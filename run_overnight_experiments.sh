#!/bin/bash
# ==========================================
# Overnight DistilBERT Fine-tuning Experiments
# Tracks GPU usage, logs output, and saves results by task
# ==========================================

# Create folders if not exist
mkdir -p logs
mkdir -p gpu_logs

echo "🚀 Starting overnight tuned experiments..."
echo "🕒 $(date)"

# Start GPU monitoring every 2 minutes
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv -l 120 > gpu_logs/gpu_usage_$(date +"%Y%m%d_%H%M%S").log &
GPU_LOG_PID=$!
echo "📊 GPU monitor started (PID: $GPU_LOG_PID)"

# ========== EMOTION ==========
echo "🔹 Running Emotion Classification (5 epochs, lr=3e-5, len=160)"
python train_models.py task emotion \
  --epochs 5 \
  --batch-size 32 \
  --learning-rate 3e-5 \
  --max-length 160 \
  --warmup-ratio 0.1 \
  --run-tag emo_opt_bs32_lr3e5_ep5_len160_wr01 \
  > logs/emotion_opt_$(date +"%Y%m%d_%H%M%S").log 2>&1

# ========== SENTIMENT ==========
echo "🔹 Running Sentiment Analysis (6 epochs, lr=2e-5, len=160)"
python train_models.py task sentiment \
  --epochs 6 \
  --batch-size 32 \
  --learning-rate 2e-5 \
  --max-length 160 \
  --warmup-ratio 0.1 \
  --run-tag sent_opt_bs32_lr2e5_ep6_len160_wr01 \
  > logs/sentiment_opt_$(date +"%Y%m%d_%H%M%S").log 2>&1

# ========== SARCASM ==========
echo "🔹 Running Sarcasm Detection (8 epochs, lr=2e-5, len=256, bs=16)"
python train_models.py task sarcasm \
  --epochs 8 \
  --batch-size 16 \
  --learning-rate 2e-5 \
  --max-length 256 \
  --warmup-ratio 0.1 \
  --run-tag sarc_opt_bs16_lr2e5_ep8_len256_wr01 \
  > logs/sarcasm_opt_$(date +"%Y%m%d_%H%M%S").log 2>&1

# ========== CLEANUP ==========
echo "🧹 Killing GPU monitor..."
kill $GPU_LOG_PID
echo "✅ All experiments finished at $(date)"
echo "📁 Check 'logs/' for training logs and 'gpu_logs/' for GPU usage data."
