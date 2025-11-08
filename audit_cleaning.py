import pandas as pd
import os
import re
import langid

# limit languages you expect (helps langid be more decisive)
langid.set_languages(['en', 'id', 'ms', 'zh', 'ta', 'fr', 'es'])

datasets = [
    {"file": "Emotion_Detection_Data.csv", "text_col": "text", "label_col": "label"},
    {"file": "Sarcasm_Data.csv", "text_col": "comment", "label_col": "label"},
    {"file": "Sentiment_Data.csv", "text_col": "text", "label_col": "target"},
]

OUTPUT_DIR = "audit_flagged/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# words you KNOW are non-English in your data – you can add to this list
NON_EN_MARKERS = (
    "nyebelin", "udh", "udah", "juga", "cape", "capek",
    "lah", "lahh", "banget", "gitu", "aja", "aku", "kamu",
)

def check_language_flags(text: str):
    """
    Stricter rule:
    - if langid says not English → non_english
    - else, if ANY token looks non-English → mixed_language
    - else → []
    """
    text = str(text).strip()
    flags = []

    # allow usernames / symbols only
    if re.match(r"^[\W_@#0-9]+$", text):
        return flags

    # detect main language
    lang, _ = langid.classify(text)
    text_lower = text.lower()

    # immediate non-English if classifier says so
    if lang != "en":
        flags.append("non_english")
        return flags

    # tokenise by words
    tokens = re.findall(r"[A-Za-z]+", text)
    # if no alphabetic tokens at all → call it non-English
    if not tokens:
        flags.append("non_english")
        return flags

    # 1) explicit marker check (Indo/Malay words)
    if any(m in text_lower for m in NON_EN_MARKERS):
        flags.append("mixed_language")
        return flags

    # 2) stricter: every token must be pure A–Z
    #    if we see something weird like 'penatnya', 'makan', 'tapi' → flag
    for tok in tokens:
        # token is alphabetic but doesn't look like typical English
        # you can make this smarter later
        if not re.match(r"^[A-Za-z]+$", tok):
            flags.append("mixed_language")
            return flags

    # passed all checks → looks fully English
    return flags


def audit_dataset(file_path, text_col, label_col):
    print(f"\n🔍 Auditing (non-destructive) {file_path}...")
    df = pd.read_csv(file_path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    original_len = len(df)

    df["reason"] = ""

    # missing / empty
    missing_mask = (
        df[text_col].isna()
        | df[label_col].isna()
        | (df[text_col].astype(str).str.strip() == "")
    )
    df.loc[missing_mask, "reason"] += "missing_text_or_label; "

    # duplicates
    dup_mask = df.duplicated(subset=[text_col], keep=False)
    df.loc[dup_mask, "reason"] += "duplicate_text; "

    # language rules
    def build_lang_reason(txt):
        flags = check_language_flags(txt)
        if not flags:
            return ""
        return "".join(f + "; " for f in flags)

    df["lang_reason"] = df[text_col].astype(str).apply(build_lang_reason)
    df.loc[df["lang_reason"] != "", "reason"] += df["lang_reason"]

    flagged_df = df[df["reason"] != ""]
    print(f"⚠️ Rows that WOULD be flagged: {len(flagged_df)} / {original_len} ({len(flagged_df)/original_len:.2%})")
    print("Top reasons:")
    print(flagged_df["reason"].value_counts().head(10))

    out_path = os.path.join(OUTPUT_DIR, f"audit_flagged_{os.path.basename(file_path)}")
    df.to_csv(out_path, index=False)
    print(f"🗂️ Saved audited (full) file to: {out_path}")


for d in datasets:
    audit_dataset(d["file"], d["text_col"], d["label_col"])
