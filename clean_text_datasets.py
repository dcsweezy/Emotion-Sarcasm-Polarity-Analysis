import pandas as pd
from langdetect import detect, DetectorFactory
from tqdm import tqdm
import re

# Fix seed for langdetect reproducibility
DetectorFactory.seed = 0

# ---------- CONFIG ----------
# List your dataset file paths here
DATASETS = {
    "sentiment": "sentiment_dataset.csv",
    "sarcasm": "sarcasm_dataset.csv",
    "emotion": "emotion_dataset.csv"
}

TEXT_COLUMN = "text"     # change if your text column has a different name
LABEL_COLUMN = "label"   # change if your label column has a different name
OUTPUT_DIR = "cleaned_datasets/"
# ----------------------------

# Helper: detect if text is English
def is_english(text):
    try:
        return detect(text) == "en"
    except:
        return False

# Helper: quick heuristic for irrelevant or corrupted text
def is_relevant(text):
    if not isinstance(text, str):
        return False
    # Remove obvious junk like only numbers, URLs, or gibberish
    if len(text.strip()) < 3:
        return False
    if re.fullmatch(r"[\W_]+", text):   # only punctuation
        return False
    if re.match(r"http[s]?://", text):  # URL only
        return False
    if re.search(r"\d{6,}", text):      # long number sequences
        return False
    return True

# Helper: check for likely mislabeled text (simple heuristic)
# e.g., text has neutral words but extreme label imbalance
def detect_mislabeled(df):
    # Optional simple rule — remove labels that appear less than 5 times
    label_counts = df[LABEL_COLUMN].value_counts()
    rare_labels = label_counts[label_counts < 5].index
    df = df[~df[LABEL_COLUMN].isin(rare_labels)]
    return df

def clean_dataset(name, path):
    print(f"\n🧼 Cleaning {name} dataset...")

    # Load CSV
    df = pd.read_csv(path)

    # Drop duplicates and NaNs
    df.drop_duplicates(subset=[TEXT_COLUMN], inplace=True)
    df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN], inplace=True)

    # Remove empty or irrelevant text
    tqdm.pandas(desc="Checking relevance")
    df = df[df[TEXT_COLUMN].progress_apply(is_relevant)]

    # Detect English-only samples
    tqdm.pandas(desc="Detecting language")
    df = df[df[TEXT_COLUMN].progress_apply(is_english)]

    # Remove obviously mislabeled samples (rare labels)
    df = detect_mislabeled(df)

    # Reset index
    df.reset_index(drop=True, inplace=True)

    # Save cleaned dataset
    output_path = f"{OUTPUT_DIR}{name}_cleaned.csv"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Cleaned {name} dataset saved to: {output_path}")
    print(f"Remaining samples: {len(df)}")

if __name__ == "__main__":
    import os
    for name, path in DATASETS.items():
        clean_dataset(name, path)
