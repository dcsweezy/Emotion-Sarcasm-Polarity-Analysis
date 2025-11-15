"""Train DistilBERT models for sentiment, sarcasm, and emotion classification tasks.

This script performs the following for each task:
* Loads the appropriate dataset from CSV.
* Applies an 80/20 stratified train/test split.
* Runs stratified K-fold cross validation on the training split.
* Fine-tunes ``distilbert-base-uncased`` using Hugging Face ``Trainer``.
* Reports precision, recall, F1-score, accuracy, and ROC-AUC metrics.

The script is designed to be reproducible and configurable through command-line arguments.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback,
)
from scipy.special import softmax

LOGGER = logging.getLogger(__name__)

# we want to be able to call: python train_models.py task sentiment
AVAILABLE_TASK_NAMES = ("sentiment", "sarcasm", "emotion")

# Sentiment dataset uses ``0`` for negative and ``4`` for positive samples. Map these
# raw values to contiguous indices for the classifier while keeping the semantic
# meaning explicit for reporting.
SENTIMENT_RAW_TO_MODEL_LABEL = {0: 0, 4: 1}
SENTIMENT_ID2LABEL = {
    0: "negative (0)",
    1: "positive (4)",
}


@dataclass
class TaskConfig:
    """Configuration for a single classification task."""

    name: str
    file_path: Path
    text_column: str
    label_column: str
    num_labels: int
    label_mapping: Optional[Dict] = None
    id2label: Optional[Dict[int, str]] = None
    drop_columns: Optional[Iterable[str]] = None
    early_stopping_patience: Optional[int] = None


@dataclass
class ExperimentConfig:
    model_name: str
    test_size: float
    k_folds: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_length: int
    seed: int
    output_dir: Path
    run_tag: str
    early_stopping: bool = False
    early_stopping_patience: int = 1


def prepare_dataframe(task: TaskConfig) -> pd.DataFrame:
    """Load a task-specific dataframe and normalise labels."""
    df = pd.read_csv(task.file_path)
    LOGGER.info("Loaded %s with %d rows", task.file_path.name, len(df))

    if task.drop_columns:
        df = df.drop(columns=list(task.drop_columns), errors="ignore")

    df = df[[task.text_column, task.label_column]].copy()
    df[task.text_column] = df[task.text_column].astype(str).str.strip()
    df[task.label_column] = pd.to_numeric(df[task.label_column], errors="coerce")
    df = df.dropna(subset=[task.text_column, task.label_column])

    # remap labels if needed (for sentiment 0/4 -> 0/1)
    if task.label_mapping is not None:
        df = df[df[task.label_column].isin(task.label_mapping.keys())]
        LOGGER.info("Applying label mapping for %s: %s", task.name, task.label_mapping)
        df[task.label_column] = df[task.label_column].map(task.label_mapping)

    df[task.label_column] = df[task.label_column].astype(int)
    return df


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer, text_column: str, max_length: int) -> Dataset:
    """Tokenize a Hugging Face dataset and prepare it for Trainer."""

    def _tokenize(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        return tokenizer(
            batch[text_column],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = dataset.map(_tokenize, batched=True)
    if "label" in tokenized.column_names:
        tokenized = tokenized.rename_column("label", "labels")

    removable_columns = [
        col
        for col in tokenized.column_names
        if col not in {"input_ids", "attention_mask", "labels"}
    ]
    tokenized = tokenized.remove_columns(removable_columns)
    tokenized.set_format(type="torch")
    return tokenized


def compute_metrics_builder(num_labels: int) -> callable:
    """Build a metric computation function capturing ``num_labels``."""

    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average="weighted",
            zero_division=0,
        )
        accuracy = accuracy_score(labels, predictions)

        try:
            if num_labels == 2:
                probabilities = softmax(logits, axis=-1)[:, 1]
                auc = roc_auc_score(labels, probabilities)
            else:
                probabilities = softmax(logits, axis=-1)
                auc = roc_auc_score(labels, probabilities, multi_class="ovr")
        except ValueError:
            auc = float("nan")

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "roc_auc": auc,
        }

    return _compute_metrics

def trainer_factory(
    task: TaskConfig,
    output_dir: Path,
    model_name: str,
    tokenizer: AutoTokenizer,
    training_config: ExperimentConfig,
    train_dataset,         
    eval_dataset,  

) -> Trainer:
    """Instantiate a Trainer for a given task and run configuration."""

    num_labels = task.num_labels
    id2label = task.id2label or {i: f"LABEL_{i}" for i in range(num_labels)}
    #use correct mapping direction
    label2id = {label: idx for idx, label in id2label.items()}
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir / task.name),
        num_train_epochs=training_config.epochs,
        per_device_train_batch_size=training_config.batch_size,
        per_device_eval_batch_size=training_config.batch_size,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        warmup_ratio=training_config.warmup_ratio,
        eval_strategy="epoch",
        logging_strategy="epoch",
        report_to=[],
    )

    callbacks = []
    if training_config.early_stopping:
        # prefer task-specific patience, else use global
        patience = task.early_stopping_patience or training_config.early_stopping_patience
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,    
        eval_dataset=eval_dataset,      
        compute_metrics=compute_metrics_builder(num_labels),
        callbacks=callbacks,
    )

    return trainer


def run_cross_validation(
    task: TaskConfig,
    config: ExperimentConfig,
    tokenizer: AutoTokenizer,
    train_df: pd.DataFrame,
) -> List[Dict[str, float]]:
    """Execute stratified K-fold cross validation and return metrics for each fold."""
    labels = train_df[task.label_column].to_numpy()
    skf = StratifiedKFold(n_splits=config.k_folds, shuffle=True, random_state=config.seed)

    fold_metrics: List[Dict[str, float]] = []
    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(train_df, labels), start=1):
        LOGGER.info("%s - Fold %d/%d", task.name, fold_idx, config.k_folds)

        train_subset = train_df.iloc[train_indices].reset_index(drop=True).rename(
            columns={task.label_column: "label"}
        )
        val_subset = train_df.iloc[val_indices].reset_index(drop=True).rename(
            columns={task.label_column: "label"}
        )

        train_split = Dataset.from_pandas(train_subset)
        val_split = Dataset.from_pandas(val_subset)

        train_split = tokenize_dataset(train_split, tokenizer, task.text_column, config.max_length)
        val_split = tokenize_dataset(val_split, tokenizer, task.text_column, config.max_length)

        trainer = trainer_factory(
            task=task,
            output_dir=config.output_dir,
            model_name=config.model_name,
            tokenizer=tokenizer,
            training_config=config,
            train_dataset=train_split,
            eval_dataset=val_split,
        )
        trainer.train()
        metrics = trainer.evaluate()
        LOGGER.info("Fold %d metrics: %s", fold_idx, metrics)
        fold_metrics.append(metrics)

    return fold_metrics


def train_final_model(
    task: TaskConfig,
    config: ExperimentConfig,
    tokenizer: AutoTokenizer,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, float]:
    """Train on the entire training split and evaluate on the hold-out test set."""

    train_subset = train_df.reset_index(drop=True).rename(columns={task.label_column: "label"})
    test_subset = test_df.reset_index(drop=True).rename(columns={task.label_column: "label"})

    train_dataset = Dataset.from_pandas(train_subset)
    test_dataset = Dataset.from_pandas(test_subset)

    train_dataset = tokenize_dataset(train_dataset, tokenizer, task.text_column, config.max_length)
    test_dataset = tokenize_dataset(test_dataset, tokenizer, task.text_column, config.max_length)

    trainer = trainer_factory(
        task=task,
        output_dir=config.output_dir,
        model_name=config.model_name,
        tokenizer=tokenizer,
        training_config=config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()
    metrics = trainer.evaluate()

    LOGGER.info("%s final test metrics: %s", task.name, metrics)
    return metrics


def summarise_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Compute the mean of each metric across folds."""

    if not metrics_list:
        return {}

    summary = {}
    keys = metrics_list[0].keys()
    for key in keys:
        values = [metrics[key] for metrics in metrics_list if key in metrics]
        summary[key] = float(np.mean(values)) if values else float("nan")
    return summary


def run_task(task: TaskConfig, config: ExperimentConfig, tokenizer: AutoTokenizer) -> None:
    """Execute cross validation and hold-out evaluation for a single task."""
    df = prepare_dataframe(task)

    train_df, test_df = train_test_split(
        df,
        test_size=config.test_size,
        stratify=df[task.label_column],
        random_state=config.seed,
    )
    LOGGER.info(
        "%s dataset split into %d training and %d testing samples",
        task.name,
        len(train_df),
        len(test_df),
    )

    fold_metrics = run_cross_validation(task, config, tokenizer, train_df)
    fold_summary = summarise_metrics(fold_metrics)
    LOGGER.info("%s cross-validation summary: %s", task.name, json.dumps(fold_summary, indent=2))

    final_metrics = train_final_model(task, config, tokenizer, train_df, test_df)

    results_dir = config.output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    task_result = {
        "task": task.name,
        "hyperparams": {
            "model_name": config.model_name,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "k_folds": config.k_folds,
            "run_tag": config.run_tag,
        },
        "cross_validation": fold_metrics,
        "cross_validation_mean": fold_summary,
        "holdout_test": final_metrics,
    }

    suffix = f"_{config.run_tag}" if config.run_tag else ""
    result_path = results_dir / f"{task.name}{suffix}_metrics.json"

    with result_path.open("w", encoding="utf-8") as file:
        json.dump(task_result, file, indent=2)
    LOGGER.info("Saved metrics to %s", result_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT for sentiment, sarcasm, and emotion classification tasks"
    )
    parser.add_argument(
        "--model-name",
        default="distilbert-base-uncased",
        help="Pretrained transformer model identifier",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Per-device batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Optimizer learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.0, help="Warmup ratio for the scheduler")
    parser.add_argument("--max-length", type=int, default=160, help="Maximum sequence length for tokenization")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion for the hold-out test split")
    parser.add_argument("--k-folds", type=int, default=5, help="Number of stratified folds for cross validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for trainer outputs and metrics",
    )

    parser.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="Optional tag to append to result filenames, e.g. 'ep3_bs32_lr3e5'",
    )

    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Enable early stopping during training",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=1,
        help="Number of evaluation calls with no improvement after which to stop",
    )

    # subcommand: python train_models.py task sentiment sarcasm
    subparsers = parser.add_subparsers(dest="command")
    task_parser = subparsers.add_parser(
        "task", help="Run one or more specific tasks instead of the entire suite"
    )
    task_parser.add_argument(
        "task_names",
        nargs="+",
        choices=AVAILABLE_TASK_NAMES,
        help="Task identifiers to execute (sentiment, sarcasm, emotion)",
    )

    return parser.parse_args()


def build_task_configs(project_root: Path) -> Dict[str, TaskConfig]:
    """Construct task configurations keyed by their public task names."""
    return {
        "sentiment": TaskConfig(
            name="sentiment",
            file_path=project_root / "Sentiment_Data_Flagless.csv",
            text_column="text",
            label_column="target",
            num_labels=2,
            label_mapping=SENTIMENT_RAW_TO_MODEL_LABEL,
            id2label=SENTIMENT_ID2LABEL,
            early_stopping_patience=1,
        ),
        "sarcasm": TaskConfig(
            name="sarcasm",
            file_path=project_root / "Sarcasm_Data_Flagless.csv",
            text_column="comment",
            label_column="label",
            num_labels=2,
            id2label={0: "not_sarcastic", 1: "sarcastic"},
            early_stopping_patience=2,
        ),
        "emotion": TaskConfig(
            name="emotion",
            file_path=project_root / "Emotion_Detection_Data_Flagless.csv",
            text_column="text",
            label_column="label",
            num_labels=6,
            drop_columns=["Unnamed: 0"],
            id2label={
                0: "sadness",
                1: "joy",
                2: "love",
                3: "anger",
                4: "fear",
                5: "surprise",
            },
            early_stopping_patience=3, 
        ),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    args = parse_args()
    set_seed(args.seed)

    config = ExperimentConfig(
        model_name=args.model_name,
        test_size=args.test_size,
        k_folds=args.k_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        seed=args.seed,
        output_dir=args.output_dir,
        run_tag=args.run_tag,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)

    project_root = Path(__file__).resolve().parent
    task_configs = build_task_configs(project_root)

    # if command is to train per task: python train_models.py task sentiment
    if args.command == "task":
        selected_task_names = getattr(args, "task_names", [])
    else:
        # default: run all
        selected_task_names = list(AVAILABLE_TASK_NAMES)

    for task_name in selected_task_names:
        task = task_configs[task_name]
        LOGGER.info("Starting task: %s", task.name)
        run_task(task, config, tokenizer)

    LOGGER.info(
        "Completed tasks: %s. Metrics stored in %s",
        ", ".join(selected_task_names),
        config.output_dir / "results",
    )


if __name__ == "__main__":
    main()