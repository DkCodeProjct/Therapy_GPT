import torch
import torch.nn as nn
import torch.nn.functional as F

batchsiz = 64
blocksiz = 128
epochs = 700
evalIntervals = 100
lr = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
evaliters = 200
nemb = 158
nhead = 4
nlayers = 4
dropout = 0.2


with open('/content/train.csv', 'r', encoding='utf-8') as f:
    txt = f.read()

# Initialize tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")

def enc(txt, tokenizer):
    tokens = tokenizer(txt, return_tensors="pt", truncation=True, padding=False)["input_ids"]
    return tokens.flatten()

data = torch.tensor(enc(txt, tokenizer), dtype=torch.long)

n = int(0.9 * len(data))  # First 90% for training, last 10% for validation
trainData = data[:n]
valData = data[n:]

print(f"Training data size: {trainData.size(0)}")
print(f"Validation data size: {valData.size(0)}")


vocabsiz = len(tokenizer)
print("vocab siz: ", vocabsiz)


def getBatch(split, block_size=128, batch_size=32):
    dataset = trainData if split == "train" else valData
    ix = torch.randint(0, len(dataset) - block_size, (batch_size,))

    x = torch.stack([dataset[i:i + block_size] for i in ix])  # Inputs
    y = torch.stack([dataset[i + 1:i + block_size + 1] for i in ix])  # Targets
    x, y = x.to(device), y.to(device)

    return x, y



@torch.no_grad()
def estimateLoss():
    out = { }
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(evaliters)
        for k in range(evaliters):
            X, Y = getBatch(split)
            logits, loss = model(X, Y)
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
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("tril", torch.tril(torch.ones(blocksiz, blocksiz)))

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
            nn.ReLU(),
            nn.Linear(4 * nemb, nemb),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, nemb, nhead):
        super().__init__()
        headsiz =  nemb // nhead
        self.selfattn = MultiHeadAttention(nhead, headsiz)
        self.ffn = FeedForwardNetwork(nemb)
        self.ln_1 = nn.LayerNorm(nemb)
        self.ln_2 = nn.LayerNorm(nemb)

    def forward(self, x):
        x = x + self.selfattn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))

        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(vocabsiz, nemb)
        self.wpe = nn.Embedding(blocksiz, nemb)
        self.block = nn.Sequential(*[Block(nemb, nhead=nhead) for _ in range(nlayers)])
        self.ln_finl = nn.LayerNorm(nemb)
        self.lm_head = nn.Linear(nemb, vocabsiz)

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
        x = self.ln_finl(x)

        logits = self.lm_head(x)

        if targt is None:
            loss = None

        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targt = targt.view(B*T)
            loss = F.cross_entropy(logits, targt)

        return logits, loss
    def generate(self, idx, max_new_tokens, tokenizer):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -blocksiz:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        # Decode the generated token indices to text
        generated_text = tokenizer.decode(idx[0].cpu().numpy().tolist(), skip_special_tokens=True)
        return generated_text

model = GPTLanguageModel()
m = model.to(device)
# Use Torch.Compinle
useCompile = False
if useCompile:
    model = torch.compile(model)
optim = torch.optim.AdamW(model.parameters(), lr=lr)

lossi = []
for i in range(epochs):
    if i % evalIntervals == 0 or i == epochs - 1:
        losses = estimateLoss()
        lossi.append(losses["val"].item())
        print(f"Step {i} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

    xb, yb = getBatch("train")
    logits, loss = model(xb, yb)

    optim.zero_grad()
    loss.backward()
    optim.step()


def saveCheckpnt(model, optimizer, epoch, loss, filepath):
    checkPnt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }
    torch.save(checkPnt, filepath)
    print(f"Checkpoint saved to {filepath}")

# Saving model checkpoint
saveCheckpnt(model, optim, epochs-1, lossi[-1], "TherapyModelTrainFinl.pth")


################
##############
###
##   FINE TUNE THE MODEL
###
##############
################

# Load the tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")

# Define parameters
batch_size = 64
block_size = 128
epochs = 10  # Fewer epochs for fine-tuning
lr = 1e-5  # Lower learning rate for fine-tuning
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load your pre-trained model
checkpoint_path = "TherapyModelTrainFinl.pth"
model = GPTLanguageModel()
model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"])
model = model.to(device)

# Load the fine-tuning dataset
with open('/content/finetune.csv', 'r', encoding='utf-8') as f:
    fine_tune_text = f.read()

# Tokenize the fine-tuning data
def encode_text(text, tokenizer):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=False)["input_ids"]
    return tokens.flatten()

fine_tune_data = encode_text(fine_tune_text, tokenizer)

# Split the data into training and validation sets
n = int(0.9 * len(fine_tune_data))  # 90% train, 10% validation
train_data = fine_tune_data[:n]
val_data = fine_tune_data[n:]

# Function to create batches
def get_batch(split, block_size=128, batch_size=32):
    dataset = train_data if split == "train" else val_data
    ix = torch.randint(0, len(dataset) - block_size, (batch_size,))
    x = torch.stack([dataset[i:i + block_size] for i in ix])
    y = torch.stack([dataset[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Fine-tune the model
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train()
    for _ in range(len(train_data) // batch_size):
        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation loss
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for _ in range(len(val_data) // batch_size):
            xb, yb = get_batch("val")
            _, loss = model(xb, yb)
            val_loss += loss.item()
    val_loss /= (len(val_data) // batch_size)
    print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}")

# Save the fine-tuned model
torch.save(model.state_dict(), "TherapyModelFineTuned.pth")
print("Fine-tuning complete. Model saved as TherapyModelFineTuned.pth")

# Chat with the fine-tuned model
def chat_with_model(prompt, model, tokenizer, max_new_tokens=50):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        generated_text = model.generate(input_ids, max_new_tokens, tokenizer)
    return generated_text

# Example chat
while True:
    prompt = input("You: ")
    if prompt.lower() in ["exit", "quit"]:
        break
    response = chat_with_model(prompt, model, tokenizer)
    print(f"Model: {response}")
