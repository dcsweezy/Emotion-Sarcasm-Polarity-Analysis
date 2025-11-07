import pandas as pd
from langdetect import detect, DetectorFactory
import os

DetectorFactory.seed = 0  # make language detection consistent

# Define dataset configurations
datasets = [
    {"file": "Emotion_Detection_Data.csv", "text_col": "text", "label_col": "label"},
    {"file": "Sarcasm_Data.csv", "text_col": "comment", "label_col": "label"},
    {"file": "Sentiment_Data.csv", "text_col": "text", "label_col": "target"},
]

OUTPUT_DIR = "cleaned_datasets/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_dataset(file_path, text_col, label_col):
    print(f"\n🔹 Cleaning {file_path}...")

    df = pd.read_csv(file_path)
    print(f"Original shape: {df.shape}")

    # Drop unnecessary unnamed index column if present
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Drop rows with missing or empty text/label
    df = df.dropna(subset=[text_col, label_col])
    df = df[df[text_col].str.strip().astype(bool)]

    # Drop duplicates
    df = df.drop_duplicates(subset=[text_col])

    # Detect and keep only English text
    def is_english(text):
        try:
            return detect(text) == "en"
        except:
            return False

    df["is_english"] = df[text_col].apply(is_english)
    df = df[df["is_english"] == True].drop(columns=["is_english"])

    # Save cleaned dataset
    cleaned_path = os.path.join(OUTPUT_DIR, f"cleaned_{os.path.basename(file_path)}")
    df.to_csv(cleaned_path, index=False)

    print(f"Cleaned shape: {df.shape}")
    print(f"✅ Saved cleaned dataset to: {cleaned_path}")

    # Display basic label stats
    print("\nLabel distribution:")
    print(df[label_col].value_counts(normalize=True))
    print("-" * 60)


# Run cleaning on all datasets
for d in datasets:
    clean_dataset(d["file"], d["text_col"], d["label_col"])
