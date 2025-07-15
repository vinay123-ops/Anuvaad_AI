import pandas as pd
from pathlib import Path
import sentencepiece as spm
from indicnlp.normalize.indic_normalize import DevanagariNormalizer
import re

def preprocess_text(text, lang="en"):
    """Preprocess text based on language."""
    if lang == "en":
        text = text.lower()
        text = re.sub(r"[“”]", "\"", text)
        text = re.sub(r"’", "'", text)
    else:
        normalizer = DevanagariNormalizer()
        text = normalizer.normalize(text)
    return text

def train_custom_tokenizer(lang_code="hi", save_dir="custom_tokenizer", vocab_size=50000):
    # Load dataset
    train_df = pd.read_csv(f"data/processed/en_{lang_code}_train.csv")

    # Preprocess
    train_df["en"] = train_df["en"].apply(lambda x: preprocess_text(str(x), lang="en"))
    train_df[lang_code] = train_df[lang_code].apply(lambda x: preprocess_text(str(x), lang=lang_code))

    # Combine English + Hindi
    all_texts = pd.concat([train_df["en"], train_df[lang_code]]).dropna().unique().tolist()

    # Write to temporary file
    temp_file = Path(save_dir) / "temp_train.txt"
    temp_file.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_file, "w", encoding="utf-8") as f:
        for line in all_texts:
            f.write(line + "\n")

    # Train tokenizer (REMOVED unigram_alpha)
    spm.SentencePieceTrainer.train(
    input=str(temp_file),
    model_prefix=str(Path(save_dir) / "spm"),
    vocab_size=vocab_size,
    model_type="unigram",
    character_coverage=0.9995,
    unk_id=3,
    pad_id=0,
    bos_id=1,
    eos_id=2,
    user_defined_symbols=["<pad>", "<s>", "</s>"]
)

    # Load and inspect tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(str(Path(save_dir) / "spm.model"))

    print("\n--- Sample Tokenizations ---")
    for i in range(min(5, len(train_df))):
        en = train_df["en"].iloc[i]
        hi = train_df[lang_code].iloc[i]
        en_tokens = sp.encode_as_pieces(en)
        hi_tokens = sp.encode_as_pieces(hi)
        print(f"\nSample {i+1} EN: {en}")
        print(f"Toks EN: {en_tokens}")
        print(f"UNK count EN: {en_tokens.count('<unk>')}")
        print(f"Sample {i+1} HI: {hi}")
        print(f"Toks HI: {hi_tokens}")
        print(f"UNK count HI: {hi_tokens.count('<unk>')}")
        print("-" * 50)

    # Vocabulary stats
    vocab_size = sp.get_piece_size()
    hindi_tokens = [sp.id_to_piece(i) for i in range(vocab_size) if any(0x0900 <= ord(c) <= 0x097F for c in sp.id_to_piece(i))]
    english_tokens = [sp.id_to_piece(i) for i in range(vocab_size) if not any(0x0900 <= ord(c) <= 0x097F for c in sp.id_to_piece(i)) and sp.id_to_piece(i) not in ["<pad>", "<s>", "</s>", "<unk>"]]
    print(f"\nVocabulary Stats:")
    print(f"Total vocab size: {vocab_size}")
    print(f"Hindi tokens in vocab: {len(hindi_tokens)}")
    print(f"Sample Hindi tokens: {hindi_tokens[:10]}")
    print(f"English tokens in vocab: {len(english_tokens)}")
    print(f"Sample English tokens: {english_tokens[:10]}")

    temp_file.unlink()
    print(f"\n✅ Tokenizer saved to {Path(save_dir)/'spm.model'}")

# Run this
train_custom_tokenizer(lang_code="hi")
