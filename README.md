# Anuvaad AI 📝 → 🪔  
### English to Hindi Translation Prototype  

A **custom-built, fully modular machine translation system** that translates English to Hindi using:
- **AI4Bharat's Samanantar dataset**
- **Custom preprocessing pipeline**
- **Custom SentencePiece tokenizer**
- **Custom-built Transformer model (PyTorch)**  

No Hugging Face model weights used — built from scratch for research and learning purposes.

---

## 🚀 Features
✅ Uses real-world aligned data (Samanantar)  
✅ Unicode normalization for Indic scripts  
✅ SentencePiece tokenizer trained from scratch  
✅ Lightweight PyTorch-based Transformer  
✅ Greedy decoding for inference  
✅ Modular, clean and extensible code  

---

## 📂 Project Structure
Anuvaad_AI/
├── data/
│ └── processed/ # Cleaned train/validation CSV
├── custom_tokenizer/
│ ├── spm.model # SentencePiece model
│ ├── spm.vocab # SentencePiece vocab
├── en_hi_transformer.pth # Saved Transformer model weights
├── src/
│ ├── data_preprocessing.py # Dataset preprocessing (Samanantar)
│ ├── tokenizer_training.py # SentencePiece tokenizer training
│ ├── model.py # Transformer model
│ └── train.py # Training loop and evaluation
├── requirements.txt
├── .gitignore
└── README.md

---

## 📊 Dataset
| Dataset      | Language Pair | Purpose       |
|--------------|---------------|---------------|
| **Samanantar** | English-Hindi | Parallel corpus for machine translation |

- Source: [AI4Bharat / Samanantar](https://huggingface.co/datasets/ai4bharat/samanantar)
- Loaded directly via 🤗 `datasets` library.

---

## 🛠 Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/vinay123-ops/Anuvaad_AI.git
cd Anuvaad_AI
2️⃣ Install Dependencies
pip install -r requirements.txt
requirements.txt
nginx
Copy
Edit
pandas
datasets
torch
sentencepiece
indic-nlp-library
```
🔧 Pipeline Overview
🔹 1️⃣ Data Preprocessing (Samanantar)
Cleans data, normalizes Indic scripts, exports CSV.
python src/data_preprocessing.py

🔹 2️⃣ Custom Tokenizer Training
Combines English + Hindi corpus, trains SentencePiece tokenizer (unigram).
python src/tokenizer_training.py

🔹 3️⃣ Transformer Model Training
Trains custom PyTorch Transformer from scratch on tokenized data.
python src/train.py

🔥 Model Architecture
Component	Detail
Type	Encoder-Decoder Transformer
Layers	4 Encoder, 4 Decoder
Embedding	512-dim
Heads	8
FF Dim	2048
Vocabulary	SentencePiece (50k)
Special Tokens	<s>, </s>, <pad>

✨ Inference Example
python
from src.model import TransformerTranslationModel, translate_sentence
test_sentence = "The prime minister addressed the nation."
print(translate_sentence(model, test_sentence))
Output:

makefile
EN: The prime minister addressed the nation.
HI: प्रधानमंत्री ने राष्ट्र को संबोधित किया।
📉 Results (Prototype Stage)
Metric	Value
BLEU	N/A (prototype only)
Train	✅ Functional
Val	✅ Functional
Output	✅ Correct on simple sentences

📝 License
MIT License.
