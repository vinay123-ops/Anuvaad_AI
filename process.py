import pandas as pd
from pathlib import Path
import unicodedata
from datasets import load_dataset

def process_and_save(lang_code, sample_size=50000, val_size=5000):  # 🔽 Reduced size
    print(f"Loading dataset for: {lang_code}")
    
    # Load dataset
    ds = load_dataset("ai4bharat/samanantar", lang_code, split="train")
    
    # Convert to DataFrame
    df = pd.DataFrame(ds)
    df = df.rename(columns={"src": "en", "tgt": lang_code})
    
    # Normalize Unicode (NFC) for Indic scripts
    df["en"] = df["en"].apply(lambda x: unicodedata.normalize("NFC", str(x)))
    df[lang_code] = df[lang_code].apply(lambda x: unicodedata.normalize("NFC", str(x)))
    
    # Drop missing/empty rows
    df.dropna(inplace=True)
    df = df[df["en"].str.strip().str.len() > 0]
    df = df[df[lang_code].str.strip().str.len() > 0]
    
    # Sample train and validation sets
    df = df.sample(n=min(sample_size + val_size, len(df)), random_state=42)
    train_df = df.iloc[:sample_size]
    val_df = df.iloc[sample_size:sample_size + val_size]
    
    # Save to CSV
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = output_dir / f"en_{lang_code}_train.csv"
    val_file = output_dir / f"en_{lang_code}_val.csv"
    train_df[["en", lang_code]].to_csv(train_file, index=False)
    val_df[["en", lang_code]].to_csv(val_file, index=False)
    
    # Validate a few samples
    print(f"Sample pairs for {lang_code}:")
    for i in range(min(3, len(train_df))):
        print(f"  EN: {train_df.iloc[i]['en']}")
        print(f"  {lang_code.upper()}: {train_df.iloc[i][lang_code]}")
    
    print(f"✅ Saved: {train_file} ({len(train_df)} pairs), {val_file} ({len(val_df)} pairs)")

def main():
    for lang in ['hi']:
        process_and_save(lang)

if __name__ == "__main__":
    main()
