import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.utils
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import os 
from train_lacan_therpyBot_model import GPT


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batchsiz = 64
blocksiz = 128
epochs = 80
lr = 1e-2

# Paths
preTrainedModel = "/kaggle/working/Therapy_bot_Trained_model.pth"
datasetPath = "/kaggle/input/your-dataset-file.txt"  # Update with your dataset file path

# Add special tokens
specialTok = {"sep_token": "<|sep|>"}
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
tokenizer.add_special_tokens(specialTok)

# Dataset Class
class TherapyData(Dataset):
    def __init__(self, filPth, tokenizer):
        with open(filPth, 'r', encoding='utf-8') as file:
            txt = file.readlines()
        
        self.examples = []
        for line in txt:
            tokLine = tokenizer.encode(line.strip(), truncation=True, max_length=blocksiz)
            self.examples.append(tokLine)
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, ix):
        return torch.tensor(self.examples[ix], dtype=torch.long)

# Load Model and Tokenizer
def loadModelAndTokenize(checkPointPath, device=device):
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
    
    # Resize embeddings to include special tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # Load checkpoint
    checkpoint = torch.load(checkPointPath, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)

    return model, tokenizer

# Fine-Tuning Function
def fineTuneModel(model, tokenizer, datasetFile):
    # Prepare dataset and dataloader
    dataset = TherapyData(datasetFile, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batchsiz,
        shuffle=True,
        collate_fn=lambda x: torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=tokenizer.pad_token_id),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, inputs in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Loss: {loss.item()}")

        print(f"Epoch {epoch + 1} completed. Average Loss: {total_loss / len(dataloader)}")
    
    # Save the fine-tuned model and tokenizer
    torch.save(model.state_dict(), "fine_tuned_model.pth")
    torch.save(tokenizer, "fine_tuned_tokenizer.pth")
    print("Model and tokenizer saved successfully.")

# Load model and fine-tune
model, tokenizer = loadModelAndTokenize(preTrainedModel, device)
fineTuneModel(model, tokenizer, datasetPath)