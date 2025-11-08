import pandas as pd
from langdetect import detect, DetectorFactory
import os

DetectorFactory.seed = 0

# same configs you used
datasets = [
    {"file": "Emotion_Detection_Data.csv", "text_col": "text", "label_col": "label"},
    {"file": "Sarcasm_Data.csv", "text_col": "comment", "label_col": "label"},
    {"file": "Sentiment_Data.csv", "text_col": "text", "label_col": "target"},
]

OUTPUT_DIR = "audit_removed/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def is_english(text):
    try:
        return detect(text) == "en"
    except:
        return False

def audit_dataset(file_path, text_col, label_col):
    print(f"\n🔍 Auditing {file_path}...")
    df = pd.read_csv(file_path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    original_len = len(df)

    # mark reasons for removal
    df["reason"] = ""

    # missing text or label
    missing_mask = df[text_col].isna() | df[label_col].isna() | (df[text_col].astype(str).str.strip() == "")
    df.loc[missing_mask, "reason"] += "missing_text_or_label; "

    # duplicate text
    dup_mask = df.duplicated(subset=[text_col], keep=False)
    df.loc[dup_mask, "reason"] += "duplicate_text; "

    # non-English text
    df["lang_is_en"] = df[text_col].astype(str).apply(is_english)
    non_en_mask = ~df["lang_is_en"]
    df.loc[non_en_mask, "reason"] += "non_english; "

    removed_df = df[df["reason"] != ""]
    kept_df = df[df["reason"] == ""]

    print(f"🧹 Removed {len(removed_df)} / {original_len} rows ({len(removed_df)/original_len:.2%})")
    print("✅ Remaining after cleaning:", len(kept_df))

    # save audit details
    audit_path = os.path.join(OUTPUT_DIR, f"audit_removed_{os.path.basename(file_path)}")
    removed_df.to_csv(audit_path, index=False)
    print(f"🗂️ Saved removed rows to: {audit_path}")

# run audits
for d in datasets:
    audit_dataset(d["file"], d["text_col"], d["label_col"])
