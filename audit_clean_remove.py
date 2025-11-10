import pandas as pd
import os
import re
import langid

# make langid a bit more decisive
langid.set_languages(['en', 'id', 'ms', 'zh', 'ta', 'fr', 'es'])

datasets = [
    {"file": "Emotion_Detection_Data.csv", "text_col": "text", "label_col": "label"},
    {"file": "Sarcasm_Data.csv", "text_col": "comment", "label_col": "label"},
    {"file": "Sentiment_Data.csv", "text_col": "text", "label_col": "target"},
]

AUDIT_DIR = "audit_flagged/"
os.makedirs(AUDIT_DIR, exist_ok=True)

# words you KNOW are non-English / non-target
NON_EN_MARKERS = (
    "nyebelin", "udh", "udah", "juga", "cape", "capek",
    "lah", "lahh", "banget", "gitu", "aja", "aku", "kamu",
)

def check_language_flags(text: str):
    """
    Return [] if ok, otherwise a list like ['non_english'] or ['mixed_language'].
    """
    text = str(text).strip()
    flags = []

    # usernames / symbols only → let it pass
    if re.match(r"^[\W_@#0-9]+$", text):
        return flags

    # langid classify
    lang, _ = langid.classify(text)
    text_lower = text.lower()

    # if main lang is not English → flag
    if lang != "en":
        flags.append("non_english")
        return flags

    # grab tokens
    tokens = re.findall(r"[A-Za-z]+", text)
    if not tokens:
        flags.append("non_english")
        return flags

    # explicit word markers for your corpus
    if any(m in text_lower for m in NON_EN_MARKERS):
        flags.append("mixed_language")
        return flags

    # extra strict: all alpha
    for tok in tokens:
        if not re.match(r"^[A-Za-z]+$", tok):
            flags.append("mixed_language")
            return flags

    return flags


def process_dataset(file_path, text_col, label_col):
    print(f"\n🔍 Auditing {file_path} ...")
    df = pd.read_csv(file_path)

    # drop unnamed cols
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    original_len = len(df)

    # create reason col
    df["reason"] = ""

    # 1) missing text/label
    missing_mask = (
        df[text_col].isna()
        | df[label_col].isna()
        | (df[text_col].astype(str).str.strip() == "")
    )
    df.loc[missing_mask, "reason"] += "missing_text_or_label; "

    # 2) duplicates
    dup_mask = df.duplicated(subset=[text_col], keep=False)
    df.loc[dup_mask, "reason"] += "duplicate_text; "

    # 3) language / mixed
    def build_lang_reason(txt):
        flags = check_language_flags(txt)
        if not flags:
            return ""
        return "".join(f + "; " for f in flags)

    df["lang_reason"] = df[text_col].astype(str).apply(build_lang_reason)
    df.loc[df["lang_reason"] != "", "reason"] += df["lang_reason"]

    # save full audited version
    audited_path = os.path.join(AUDIT_DIR, f"audit_flagged_{os.path.basename(file_path)}")
    df.to_csv(audited_path, index=False)
    print(f"🗂️ Saved audited (with reasons) to: {audited_path}")

    # now create flagless/clean version: keep only rows with no reason
    clean_df = df[df["reason"] == ""].copy()
    cleaned_name = os.path.splitext(os.path.basename(file_path))[0] + "_Flagless.csv"
    clean_df.to_csv(cleaned_name, index=False)

    print(f"⚠️ Flagged rows: {original_len - len(clean_df)} / {original_len} "
          f"({(original_len - len(clean_df)) / original_len:.2%})")
    print(f"✅ Cleaned dataset saved to: {cleaned_name}")


for d in datasets:
    process_dataset(d["file"], d["text_col"], d["label_col"])
