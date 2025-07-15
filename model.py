import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import math
from sacrebleu import corpus_bleu
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set device
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
device = get_device()

# Validate file existence
def load_file(file_path):
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")
    return file_path

# Load SentencePiece tokenizer
sp = spm.SentencePieceProcessor()
sp.load(load_file("custom_tokenizer/spm.model"))
VOCAB_SIZE = sp.get_piece_size()

# Dataset
class TranslationDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=64):
        self.df = pd.read_csv(load_file(csv_path))
        self.validate_csv()
        self.sp = tokenizer
        self.max_len = max_len

    def validate_csv(self):
        required_columns = ["en", "hi"]
        if not all(col in self.df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")

    def __len__(self):
        return len(self.df)

    def encode(self, text):
        ids = self.sp.encode_as_ids(str(text))
        ids = ids[:self.max_len - 2]
        return [1] + ids + [2]  # <s> ... </s>

    def pad(self, ids):
        ids = ids[:self.max_len]
        return ids + [0] * (self.max_len - len(ids))

    def __getitem__(self, idx):
        en = self.df.iloc[idx]["en"]
        hi = self.df.iloc[idx]["hi"]
        en_ids = self.pad(self.encode(en))
        hi_ids = self.pad(self.encode(hi))
        return {
            "encoder_input": torch.tensor(en_ids, dtype=torch.long),
            "decoder_input": torch.tensor(hi_ids[:-1], dtype=torch.long),
            "target": torch.tensor(hi_ids[1:], dtype=torch.long)
        }

# Load datasets
train_data = TranslationDataset("data/processed/en_hi_train.csv", sp)
val_data = TranslationDataset("data/processed/en_hi_val.csv", sp)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64)

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Transformer Model
class TransformerTranslationModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=4, dim_ff=2048, dropout=0.1, max_len=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones((sz, sz)) * float("-inf"), diagonal=1)

    def forward(self, src, tgt):
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        src_pad_mask = (src == 0)
        tgt_pad_mask = (tgt == 0)
        src_emb = self.pos_encoding(self.embedding(src))
        tgt_emb = self.pos_encoding(self.embedding(tgt))
        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask
        )
        return self.fc_out(output)

# Initialize
model = TransformerTranslationModel(vocab_size=VOCAB_SIZE).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

# Training and Evaluation
def train_epoch(model, loader):
    model.train()
    total_loss = 0
    for batch in loader:
        src = batch["encoder_input"].to(device)
        tgt = batch["decoder_input"].to(device)
        target = batch["target"].to(device)
        output = model(src, tgt)
        output = output.view(-1, VOCAB_SIZE)
        target = target.view(-1)
        loss = F.cross_entropy(output, target, ignore_index=0)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    total_loss = 0
    predictions, references = [], []
    with torch.no_grad():
        for batch in loader:
            src = batch["encoder_input"].to(device)
            tgt = batch["decoder_input"].to(device)
            target = batch["target"].to(device)
            output = model(src, tgt)
            output = output.view(-1, VOCAB_SIZE)
            target = target.view(-1)
            loss = F.cross_entropy(output, target, ignore_index=0)
            total_loss += loss.item()
            pred_ids = output.argmax(dim=-1).view(src.size(0), -1).cpu().numpy()
            predictions.extend([sp.decode_ids([i for i in ids if i not in [0, 1, 2]]) for ids in pred_ids])
            references.extend([sp.decode_ids([i for i in ids if i not in [0, 1, 2]]) for ids in batch["target"].cpu().numpy()])
    bleu = corpus_bleu(predictions, [references]).score
    return total_loss / len(loader), bleu

# Train Loop with Early Stopping
EPOCHS = 20
best_val_loss = float("inf")
patience = 3
counter = 0
for epoch in range(1, EPOCHS + 1):
    train_loss = train_epoch(model, train_loader)
    val_loss, val_bleu = evaluate(model, val_loader)
    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val BLEU = {val_bleu:.2f}")
    scheduler.step(val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "en_hi_transformer_best.pth")
        counter = 0
    else:
        counter += 1
    if counter >= patience:
        print("Early stopping triggered.")
        break

# Load best model
model.load_state_dict(torch.load("en_hi_transformer_best.pth"))
model.eval()

# Inference: Greedy decoding
def translate_sentence(model, sentence, max_len=64):
    if not sentence.strip():
        return "Input sentence is empty."
    model.eval()
    with torch.no_grad():
        encoded = sp.encode_as_ids(sentence)
        src = [1] + encoded[:max_len - 2] + [2]
        src_tensor = torch.tensor([src + [0] * (max_len - len(src))]).to(device)
        tgt_ids = [1]
        for _ in range(max_len):
            tgt_tensor = torch.tensor([tgt_ids]).to(device)
            output = model(src_tensor, tgt_tensor)
            next_token = output[0, len(tgt_ids)-1].argmax().item()
            if next_token == 2 or next_token == 0:
                break
            tgt_ids.append(next_token)
        return sp.decode_ids([i for i in tgt_ids if i not in [0, 1, 2]])

# Try a test sentence
test_sentence = "The prime minister addressed the nation."
translation = translate_sentence(model, test_sentence)
print("Translated:", translation)