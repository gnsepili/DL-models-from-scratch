import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
import wandb
import pickle
from tqdm import tqdm

from config import *
from data.dataset import ShakespeareDataset
from model.rnn import RNN
from utils.tokenizer import CharacterTokenizer

def train_model(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, device: torch.device) -> float:
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc="Training"):
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def train_shakespeare_model():
    wandb.init(project="rnn-shakespeare", config={
        "batch_size": BATCH_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
    })

    dataset = load_dataset("Trelis/tiny-shakespeare", split="train")
    text = dataset["Text"][0]
    tokenizer = CharacterTokenizer(text)

    train_dataset = ShakespeareDataset(text, SEQUENCE_LENGTH, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = RNN(tokenizer.vocab_size, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        avg_loss = train_model(model, train_loader, optimizer, criterion, DEVICE)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "loss": avg_loss})

    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': tokenizer.vocab_size,
    }, MODEL_PATH)
    
    with open(TOKENIZER_PATH, 'wb') as f:
        pickle.dump(tokenizer, f)
    
    print("Model and tokenizer saved successfully.")
    wandb.finish()

if __name__ == "__main__":
    train_shakespeare_model()