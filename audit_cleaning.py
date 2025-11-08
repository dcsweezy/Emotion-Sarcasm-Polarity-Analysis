import pandas as pd
from langdetect import detect, DetectorFactory
import langid

import os
import re

DetectorFactory.seed = 0

# datasets to audit (same as before)
datasets = [
    {"file": "Emotion_Detection_Data.csv", "text_col": "text", "label_col": "label"},
    {"file": "Sarcasm_Data.csv", "text_col": "comment", "label_col": "label"},
    {"file": "Sentiment_Data.csv", "text_col": "text", "label_col": "target"},
]

OUTPUT_DIR = "audit_flagged/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def check_language_flags(text: str):
    """
    Returns flags like:
    [] → good English
    ['non_english'] → clearly not English
    ['mixed_language'] → contains both English and non-English words
    """
    text = str(text).strip()
    flags = []

    # 1) emoji / username / symbols only → OK
    if re.match(r"^[\W_@#0-9]+$", text):
        return flags

    # 2) extract words
    tokens = re.findall(r"[A-Za-z]+", text)
    if len(tokens) == 0:
        flags.append("non_english")
        return flags

    # 3) detect language
    lang, _ = langid.classify(text)

    # 4) measure English-ness ratio
    english_like = [t for t in tokens if re.match(r"^[A-Za-z]+$", t)]
    english_ratio = len(english_like) / len(tokens)

    # rule logic
    if lang != "en" and english_ratio < 0.9:
        flags.append("non_english")
    elif 0.3 < english_ratio < 0.9:
        flags.append("mixed_language")

    return flags


def audit_dataset(file_path, text_col, label_col):
    print(f"\n🔍 Auditing (non-destructive) {file_path}...")
    df = pd.read_csv(file_path)

    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    original_len = len(df)

    df["reason"] = ""

    # missing / empty text or label
    missing_mask = (
        df[text_col].isna()
        | df[label_col].isna()
        | (df[text_col].astype(str).str.strip() == "")
    )
    df.loc[missing_mask, "reason"] += "missing_text_or_label; "

    # duplicate text
    dup_mask = df.duplicated(subset=[text_col], keep=False)
    df.loc[dup_mask, "reason"] += "duplicate_text; "

    # language / mixed-language
    def build_lang_reason(txt):
        flags = check_language_flags(txt)
        if not flags:
            return ""
        return "".join(f + "; " for f in flags)

    df["lang_reason"] = df[text_col].astype(str).apply(build_lang_reason)
    df.loc[df["lang_reason"] != "", "reason"] += df["lang_reason"]

    # summary
    flagged_df = df[df["reason"] != ""]
    print(f"⚠️ Rows that WOULD be flagged: {len(flagged_df)} / {original_len} ({len(flagged_df)/original_len:.2%})")
    print("Top reasons:")
    print(flagged_df["reason"].value_counts().head(10))

    # save full file
    out_path = os.path.join(OUTPUT_DIR, f"audit_flagged_{os.path.basename(file_path)}")
    df.to_csv(out_path, index=False)
    print(f"🗂️ Saved audited (full) file to: {out_path}")


for d in datasets:
    audit_dataset(d["file"], d["text_col"], d["label_col"])
