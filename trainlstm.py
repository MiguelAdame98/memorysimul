import torch
import torch.nn as nn
import re 
from datasets import load_dataset
from transformers import AutoTokenizer
from collections import Counter
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast
from model import LSTMTextModel

import os
import pickle


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_length = 70
batch_size = 25
embed_size = 300
hidden_size = 500
num_layers = 3
learning_rate = 0.005
num_epochs = 1

# Load WikiText-2 dataset using Hugging Face
def load_hf_dataset(max_vocab_size=10000):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased",clean_text=True,handle_chinese_chars=False, model_max_length=512)  # Truncate to 512
    vocab = build_vocab(dataset["train"], tokenizer, max_vocab_size)
    return dataset, vocab, tokenizer

def build_vocab(dataset, tokenizer, max_vocab_size):
    token_counts = Counter()

    # Batch processing for better performance
    batch_texts = [preprocess_text(line) for line in dataset["text"]]
    tokenized_output = tokenizer.batch_encode_plus(batch_texts, add_special_tokens=False)

    for tokens in tokenized_output["input_ids"]:
        token_counts.update(tokens)

    most_common = token_counts.most_common(max_vocab_size)
    vocab = {token: idx for idx, (token, _) in enumerate(most_common)}
    vocab["<unk>"] = len(vocab)  # Add unknown token
    return vocab

# Preprocess text during tokenization
def preprocess_text(line):
    line=line.lower().strip()
    line= re.sub(r"[^a-zA-z0-9\s']","", line)
    return line  # Lowercase and remove extra spaces

# Dataset class for Hugging Face text data
class TextDataset(Dataset):
    def __init__(self, data, vocab, tokenizer, seq_length, max_length=500):
        self.data = self.tokenize_and_encode(data, vocab, tokenizer, max_length)
        self.seq_length = seq_length

    def tokenize_and_encode(self, data, vocab, tokenizer, max_length):
        tokens = [
            vocab.get(token, vocab["<unk>"])
            for line in data["text"]
            for token in tokenizer.tokenize(preprocess_text(line))[:max_length]  # Explicit truncation
        ]
        return torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        return (
            self.data[idx:idx + self.seq_length],
            self.data[idx + 1:idx + self.seq_length + 1],
        )

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable parallelism warnings
    # Load dataset
    dataset, vocab, tokenizer = load_hf_dataset(max_vocab_size=10000)
# Define a regex pattern to keep only valid tokens (alphanumeric + spaces)
    valid_token_pattern = re.compile(r"^[a-zA-Z0-9\s]+$")

# Clean the vocabulary
    filtered_vocab = {token: idx for token, idx in vocab.items() if valid_token_pattern.match(token)}
    print(f"Filtered Vocabulary Size: {len(filtered_vocab)}")

# Save the cleaned vocabulary back
    with open("filtered_vocab.pkl", "wb") as f:
        pickle.dump(filtered_vocab, f)

    #with open("vocab.pkl", "wb") as f:
        #pickle.dump(vocab, f)
    train_dataset = TextDataset(dataset["train"], vocab, tokenizer, seq_length)

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers = min(4, os.cpu_count(),
        persistent_workers=True) 
    )
    print(len(train_loader))
    # Model setup
    vocab_size = len(vocab)
    model = LSTMTextModel(vocab_size, embed_size=300, hidden_size=512, num_layers=3, dropout=0.5).to(device)
    if torch.__version__ >= "2.0":
        model = torch.compile(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    scaler = torch.amp.GradScaler(enabled=(device.type in ["cuda", "mps"]))

def train_model():
    model.train()
    start_epoch = 0  # Track where to resume from
    checkpoint_path = "lstm_checkpoint.pth"

    # Load checkpoint if available
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1  # Resume from the next epoch
        print(f"Resumed from checkpoint, starting from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            hidden = model.init_hidden(inputs.size(0)) 
            hidden = tuple(h.detach() for h in hidden)

            optimizer.zero_grad()
            with autocast(device_type=device.type):
                outputs, hidden = model(inputs, hidden)
                loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            if batch_idx % 1000 == 0 and batch_idx != 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}")
        scheduler.step()

        # Save checkpoint after every epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "vocab_size": vocab_size
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1}")

train_model()


torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "embed_size": embed_size,
    "hidden_size": hidden_size,
    "num_layers": num_layers,
    "vocab_size": vocab_size
}, "lstm_model8.pth")
print("Model saved successfully!")