#!/bin/bash

# ===============================
# Run three NLP experiments sequentially (no logs)
# ===============================

mkdir -p outputs/results

echo "🚀 Starting sequential experiments..."
echo "🕒 $(date)"
echo "====================================="

# 1️⃣ Emotion Classification
echo "🔹 Running Emotion Classification..."
python train_models.py \
  --epochs 5 \
  --batch-size 32 \
  --learning-rate 3e-5 \
  --max-length 160 \
  --warmup-ratio 0.1 \
  --run-tag emo_opt_bs32_lr3e5_ep5_len160_wr01 \
  task emotion

# 2️⃣ Sentiment Analysis
echo "🔹 Running Sentiment Analysis..."
python train_models.py \
  --epochs 6 \
  --batch-size 32 \
  --learning-rate 2e-5 \
  --max-length 160 \
  --warmup-ratio 0.1 \
  --run-tag sent_opt_bs32_lr2e5_ep6_len160_wr01 \
  task sentiment

# 3️⃣ Sarcasm Detection
echo "🔹 Running Sarcasm Detection..."
python train_models.py \
  --epochs 8 \
  --batch-size 16 \
  --learning-rate 2e-5 \
  --max-length 256 \
  --warmup-ratio 0.1 \
  --run-tag sarc_opt_bs16_lr2e5_ep8_len256_wr01 \
  task sarcasm

echo "✅ All experiments finished at $(date)"
echo "📁 Results saved to: outputs/results/"
