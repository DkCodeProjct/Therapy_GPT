import torch
import torch.nn as nn 
import torch.nn.functional as F 
from transformers import AutoTokenizer
import pandas as pd 

with open("/kaggle/input/200232823/train.csv", 'r', encoding='utf-8') as file:
    txt = file.read()

# hyper para
blocksiz = 128
batchsiz = 64
epochs = 700
evalIntervals = 200
evaliters = 50
nemb = 42
nhead = 1
nlayers = 1
dropout = 0.0 
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 1e-2

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")

specialTok = {
    "sep_token" : "<|sep|>"
}
# tokenizer already have eos and pad
tokenizer.add_special_tokens(specialTok)

vocabsiz = tokenizer.vocab_size
print(vocabsiz)

def enc(txt, tokenizer):
    tokens = tokenizer(
        txt,
        return_tensors="pt", 
        truncation=True,
        padding=True,
        add_special_tokens=True
    )["input_ids"]

    return tokens.flatten()

data = enc(txt, tokenizer)
n = int(0.9*len(data))
trainData = data[:n]
valData = data[n:]

trainData = torch.tensor(trainData, dtype=torch.long)
valData = torch.tensor(valData, dtype=torch.long)
print(f"Train Data Shape: {trainData.shape}")
print(f"Validation Data Shape: {valData.shape}")

def getBatch(split):
    data = trainData if split == "train" else valData
    ix = torch.randint(0, len(data) - blocksiz, (batchsiz,))
    x = torch.stack([data[i:i+blocksiz] for i in ix])
    y = torch.stack([data[i+1:i+blocksiz+1] for i in ix])
    x, y = x.to(device), y.to(device)

    return x, y 

@torch.no_grad()
def estimateLoss():
    out = { }
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(evaliters)
        for k in range(evaliters):
            x, y = getBatch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    model.train()
    return out 

class Head(nn.Module):
    def __init__(self, headsiz):
        super().__init__()
        self.key = nn.Linear(nemb, headsiz, bias=False)
        self.quary = nn.Linear(nemb, headsiz, bias=False)
        self.value = nn.Linear(nemb, headsiz, bias=False)
        
        self.register_buffer("tril", torch.tril(torch.ones(blocksiz, blocksiz)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.quary(x)

        w = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        w = F.softmax(w, dim=-1)
        w = self.dropout(w)
        
        v = self.value(x)
        out = w @ v 
        return out 

class MultiHeadAttention(nn.Module):
    def __init__(self, nhead, headsiz):
        super().__init__()
        self.heads = nn.ModuleList([Head(headsiz) for _ in range(nhead)])
        self.proj = nn.Linear(headsiz * nhead, nemb)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out 

class FeedForwardNetwork(nn.Module):
    def __init__(self, nemb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nemb, 4 * nemb), 
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * nemb, nemb), 
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, nemb, nhead):
        super().__init__()
        headsiz = nemb // nhead
        self.sa = MultiHeadAttention(nhead, headsiz)
        self.ffn = FeedForwardNetwork(nemb)
        self.ln_1 = nn.LayerNorm(nemb)
        self.ln_2 = nn.LayerNorm(nemb)  
    
    def forward(self, x):
        x = x + self.sa(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x 


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        newVocabSiz = len(tokenizer)
        self.wte = nn.Embedding(newVocabSiz, nemb)
        torch.nn.init.normal_(model.wte.weight, mean=0.0, std=0.02)

        self.wpe = nn.Embedding(blocksiz, nemb)
        self.block = nn.Sequential(*[Block(nemb, nhead=nhead) for _ in range(nlayers)])
        self.ln_f = nn.LayerNorm(nemb)
        
        self.lm_head = nn.Linear(nemb, vocabsiz)
        torch.nn.init.normal_(model.lm_head.weight, mean=0.0, std=0.02)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ix, targt=None):
        B, T = ix.shape
        tokEmb = self.wte(ix)
        posEmb = self.wpe(torch.arange(T, device=device))

        x = tokEmb + posEmb
        x = self.block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targt is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targt = targt.view(B * T)
            loss = F.cross_entropy(logits, targt)
        return logits, loss 

########### Truncate generally means to shorten something by cutting off a portion of it, ####

    def generate(self, ix, max_new_tokens, tokenizer, tempertaure=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # Truncate input to the context window size
            ix_cond = ix[:, -min(blocksiz, ix.shape[1]):]
            logits, _ = self(ix_cond)
            logits = logits[:, -1, :]

            if tempertaure != 1.0:
                logits = logits / tempertaure
            
            if top_k is not None :
                val, indeces = torch.topk(logits, k=top_k, dim=-1)
                logits = torch.zeros_like(logits).scatter_(-1, indeces, val)
            
            probs = F.softmax(logits, dim=-1)

            ixNxt = torch.multinomial(probs, num_samples=1)
            ix = torch.cat((ix, ixNxt), dim=-1)
        
        return tokenizer.decode(ix[0].cpu().numpy().tolist(), skip_special_tokens=True)
        

            
# after add token all this line, [when init model]
newVocabSiz = len(tokenizer)
model = GPT()
model.wte = nn.Embedding(newVocabSiz, nemb)
torch.nn.init.normal_(model.wte.weight, mean=0.0, std=0.02)
m = model.to(device)


# Use compile
useCompile = False 
if useCompile:
    model = torch.compile(model)
    print("Using compile")
else:
    print("not using Compile")

#--------------
# save checkpoint path
def saveCheckpoint(model, optim, epoch, loss, file):
    chekPnt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "epoch": epoch,
        "loss": loss,
        
    }   
    torch.save(chekPnt, file)
#--------------


# Add Mixed Precision
from torch.cuda.amp import GradScaler, autocast
scaler = GradScaler()

optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
# lr schedular
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

#ploting loss
traini=[ ]
trainloss_i=[]
valloss_i=[]

for i in range(epochs):
    if i % evaliters == 0 or i == epochs - 1:
        losses = estimateLoss()
        trainloss_i.append(losses["train"].item())
        valloss_i.append(losses["val"].item())
        traini.append(i)
        
        print(f"Epoch {i}/{epochs} | Train loss {losses['train']:.4f} | Val loss {losses['val']:.4f}")
    
    xb, yb = getBatch("train")
    with autocast():
        logits, loss = model(xb, yb)

    scaler.scale(loss).backward()
    scaler.step(optim)
    scaler.update()

    optim.zero_grad(set_to_none=True)

    saveIntervals = 50 if i <= 200 else 400
    if i % saveIntervals == 0 or i == epochs - 1:
        saveCheckpoint(
            model,
            optim,
            i,
            valloss_i[-1] if valloss_i else float("inf"), 
            "Therapy_bot_Trained_model.pth"
        ) 


#############
#############
#############
# PLOT LOSS
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(traini, trainloss_i, label="train loss")
plt.plot(traini, valloss_i, label="val loss")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title("train/val loss")
plt.show()

