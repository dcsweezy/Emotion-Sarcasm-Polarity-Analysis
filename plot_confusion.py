import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_data(csv_path, text_col, label_col, sentiment_raw):
    df = pd.read_csv(csv_path)
    df = df[[text_col, label_col]].copy()

    df[text_col] = df[text_col].astype(str).str.strip()
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df = df.dropna(subset=[text_col, label_col])

    # Optional: map sentiment raw labels 0/4 -> 0/1
    if sentiment_raw:
        mapping = {0: 0, 4: 1}
        df = df[df[label_col].isin(mapping.keys())]
        df[label_col] = df[label_col].map(mapping)

    df[label_col] = df[label_col].astype(int)
    return df


def build_test_split(df, label_col, test_size=0.2, seed=42):
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_col],
        random_state=seed,
    )
    return test_df.reset_index(drop=True)


def get_predictions(
    model_dir: Path,
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    max_length: int,
    batch_size: int = 32,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].to_numpy()

    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    dataset = torch.utils.data.TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"],
        torch.tensor(labels),
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, batch_labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    return all_labels, all_preds


def plot_confusion_matrix(cm, labels, out_path, normalize=False, title="Confusion matrix"):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:.2f}" if normalize else int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate confusion matrix from a trained checkpoint.")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to model checkpoint directory.")
    parser.add_argument("--csv", type=str, required=True, help="Path to dataset CSV.")
    parser.add_argument("--text-col", type=str, required=True, help="Name of the text column.")
    parser.add_argument("--label-col", type=str, required=True, help="Name of the label column.")
    parser.add_argument("--labels", type=str, required=True, help="Comma-separated label names in index order.")
    parser.add_argument("--max-length", type=int, default=160, help="Max token length (same as training).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size (must match training).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (must match training).")
    parser.add_argument("--out-name", type=str, required=True, help="Path to output PNG file.")
    parser.add_argument(
        "--sentiment-raw",
        action="store_true",
        help="Use this if labels are 0/4 for sentiment and need mapping to 0/1.",
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    csv_path = Path(args.csv)
    label_names = [x.strip() for x in args.labels.split(",")]

    # 1) load & prep data
    df = load_data(csv_path, args.text_col, args.label_col, sentiment_raw=args.sentiment_raw)
    test_df = build_test_split(df, args.label_col, test_size=args.test_size, seed=args.seed)

    # 2) get predictions on test split
    y_true, y_pred = get_predictions(
        model_dir=model_dir,
        df=test_df,
        text_col=args.text_col,
        label_col=args.label_col,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    # 3) confusion matrix + classification report
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)
    print("\nClassification report:\n")
    print(classification_report(y_true, y_pred, target_names=label_names, digits=4))

    # 4) save confusion matrix (both raw + normalized)
    base_out = Path(args.out_name)
    plot_confusion_matrix(
        cm,
        label_names,
        out_path=base_out,
        normalize=False,
        title="Confusion matrix (counts)",
    )
    plot_confusion_matrix(
        cm,
        label_names,
        out_path=base_out.with_name(base_out.stem + "_norm.png"),
        normalize=True,
        title="Confusion matrix (normalized)",
    )

    print(f"\n✅ Saved confusion matrices to: {base_out} and {base_out.with_name(base_out.stem + '_norm.png')}")


if __name__ == "__main__":
    main()
