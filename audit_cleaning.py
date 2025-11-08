import pandas as pd
from langdetect import detect, DetectorFactory
import os

DetectorFactory.seed = 0

# datasets to audit (same as before)
datasets = [
    {"file": "Emotion_Detection_Data.csv", "text_col": "text", "label_col": "label"},
    {"file": "Sarcasm_Data.csv", "text_col": "comment", "label_col": "label"},
    {"file": "Sentiment_Data.csv", "text_col": "text", "label_col": "target"},
]

OUTPUT_DIR = "audit_flagged/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except:
        return False

def audit_dataset(file_path, text_col, label_col):
    print(f"\n🔍 Auditing (non-destructive) {file_path}...")
    df = pd.read_csv(file_path)

    # drop unnamed columns from display, but we won't drop rows
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    original_len = len(df)

    # add columns for auditing
    df["reason"] = ""
    df["lang_is_en"] = True  # default

    # 1) missing / empty text or label
    missing_mask = df[text_col].isna() | df[label_col].isna() | (df[text_col].astype(str).str.strip() == "")
    df.loc[missing_mask, "reason"] += "missing_text_or_label; "

    # 2) duplicate text (keep=False → mark all duplicates)
    dup_mask = df.duplicated(subset=[text_col], keep=False)
    df.loc[dup_mask, "reason"] += "duplicate_text; "

    # 3) language check
    df["lang_is_en"] = df[text_col].astype(str).apply(is_english)
    non_en_mask = ~df["lang_is_en"]
    df.loc[non_en_mask, "reason"] += "non_english; "

    # summary
    flagged_df = df[df["reason"] != ""]
    print(f"⚠️ Rows that WOULD be removed: {len(flagged_df)} / {original_len} ({len(flagged_df)/original_len:.2%})")
    print("Top reasons:")
    print(flagged_df["reason"].value_counts().head(10))

    # save full dataset with reasons (nothing removed)
    out_path = os.path.join(OUTPUT_DIR, f"audit_flagged_{os.path.basename(file_path)}")
    df.to_csv(out_path, index=False)
    print(f"🗂️ Saved audited (full) file to: {out_path}")

for d in datasets:
    audit_dataset(d["file"], d["text_col"], d["label_col"])
