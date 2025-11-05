# Emotion, Sarcasm, and Sentiment Analysis with DistilBERT

This project fine-tunes the [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased) transformer model to
explore how emotional polarity and sarcasm can co-occur in social media text. The training pipeline covers three
classification tasks:

1. **Binary Sentiment Classification** – Distinguish positive vs. negative sentiment using `Sentiment_Data.csv`.
2. **Sarcasm Detection** – Identify sarcastic vs. non-sarcastic utterances using `Sarcasm_Data.csv`.
3. **Emotion Labelling** – Predict one of six discrete emotions (sadness, anger, joy, love, surprise, fear) using
   `Emotion_Detection_Data.csv`.

Each task uses an 80/20 stratified train/test split, stratified K-fold cross-validation on the training set, and reports
Precision, Recall, F1-Score, Accuracy, and ROC-AUC metrics.

## Repository Structure

```
Emotion-Sarcasm-Polarity-Analysis/
├── Emotion_Detection_Data.csv
├── Sarcasm_Data.csv
├── Sentiment_Data.csv
├── README.md
├── requirements.txt
└── train_models.py
```

## Prerequisites

- Python 3.9 or newer
- Git (optional, for cloning the repository)
- A machine with a CUDA-capable GPU is highly recommended for reasonable training times. CPU-only execution is possible
  but will be significantly slower.

## Installation

1. **Clone the repository (or download the source code).**

   ```bash
   git clone https://github.com/<your-account>/Emotion-Sarcasm-Polarity-Analysis.git
   cd Emotion-Sarcasm-Polarity-Analysis
   ```

2. **(Optional) Create and activate a virtual environment.**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux / macOS
   # .venv\Scripts\activate  # Windows PowerShell
   ```

3. **Install the Python dependencies.**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   The requirements include PyTorch, Hugging Face Transformers, Datasets, and scikit-learn for model training and
   evaluation.

## Running the Training Pipeline

The `train_models.py` script orchestrates data preparation, cross-validation, fine-tuning, and evaluation for all three
classification tasks. By default it will:

- Load each dataset from CSV.
- Perform an 80/20 stratified split into train and test sets.
- Run stratified K-fold cross-validation (default 5 folds) on the training data.
- Fine-tune DistilBERT for the specified number of epochs (default 1).
- Evaluate Precision, Recall, F1-score, Accuracy, and ROC-AUC for each fold and on the held-out test set.
- Save metrics to JSON files under `outputs/results/`.

Launch the pipeline with the default configuration:

```bash
python train_models.py
```

### Command-line Options

`train_models.py` exposes several arguments for experimentation:

| Argument | Description | Default |
| --- | --- | --- |
| `--model-name` | Hugging Face model checkpoint to fine-tune. | `distilbert-base-uncased` |
| `--epochs` | Number of training epochs for each run. | `1` |
| `--batch-size` | Per-device batch size for training and evaluation. | `16` |
| `--learning-rate` | AdamW optimizer learning rate. | `5e-5` |
| `--weight-decay` | Weight decay applied during training. | `0.01` |
| `--warmup-ratio` | Scheduler warmup ratio. | `0.0` |
| `--max-length` | Maximum token length during tokenization. | `160` |
| `--test-size` | Proportion reserved for the hold-out test split. | `0.2` |
| `--k-folds` | Number of stratified folds for cross-validation. | `5` |
| `--seed` | Random seed for reproducibility. | `42` |
| `--output-dir` | Directory for Trainer artifacts and metric JSON files. | `outputs/` |

Example: train for three epochs with a smaller batch size and outputs stored in a custom directory.

```bash
python train_models.py --epochs 3 --batch-size 8 --output-dir my_experiment
```

## Outputs

After a successful run you will find:

- `outputs/<task>_fold_*` – Intermediate Trainer logs and checkpoints for each cross-validation fold.
- `outputs/<task>_final` – Artifacts from the final model trained on the full training split.
- `outputs/results/<task>_metrics.json` – Precision, Recall, F1-score, Accuracy, and ROC-AUC for every fold plus the
  hold-out test set.

Use these JSON files to compare performance across tasks or aggregate metrics across folds.

## Notes on the Datasets

- **Sentiment Analysis** – The original dataset encodes positive sentiment as `4` and negative as `0`. The script maps
  these to contiguous labels internally (`4 -> 1`, `0 -> 0`) while reporting the original values in the label names.
- **Sarcasm Detection** – The `label` column is already binary (`1` for sarcastic, `0` for non-sarcastic).
- **Emotion Classification** – The dataset uses `0`-`5` to represent `sadness (0)`, `joy (1)`, `love (2)`, `anger (3)`,
  `fear (4)`, and `surprise (5)`.

Ensure the CSV files remain in the repository root so the script can locate them.

## Reproducibility

The script sets a global seed for PyTorch, NumPy, and Python where possible. Variations can still occur due to GPU
non-determinism. For fully deterministic runs, consult the PyTorch reproducibility documentation and configure
appropriate environment variables for your hardware.

## Troubleshooting

- **CUDA out of memory:** Reduce `--batch-size` or `--max-length`, or switch to CPU by setting the `CUDA_VISIBLE_DEVICES`
  environment variable to an empty string.
- **Slow training:** Consider using a GPU-enabled environment or limiting the number of epochs and folds.
- **Metric calculation errors:** ROC-AUC may be undefined for heavily imbalanced folds; in those cases the score will be
  reported as `NaN`.

## License

This repository is provided for educational purposes. Refer to the datasets' respective licenses before redistributing
or deploying models trained on them.
