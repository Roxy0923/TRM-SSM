#!/bin/bash
# EEMFlow Training Script with organized logging
# Usage: ./start_training.sh <batch_size> <lr> <train_iters> <val_iters>

BATCH_SIZE=${1:-96}
LR=${2:-3.2e-4}
WD=${3:-5e-5}
TRAIN_ITERS=${4:-600000}
VAL_ITERS=${5:-1000}
NUM_WORKERS=${6:-8}

echo "=================================================="
echo "EEMFlow Training Configuration"
echo "=================================================="
echo "Batch Size:      $BATCH_SIZE"
echo "Learning Rate:   $LR"
echo "Weight Decay:    $WD"
echo "Train Iterations: $TRAIN_ITERS"
echo "Val Interval:    $VAL_ITERS"
echo "Num Workers:     $NUM_WORKERS"
echo "=================================================="
echo ""

# Start training
CUDA_VISIBLE_DEVICES=0,1 conda run -n events_signals python train_EEMFlow_HREM.py \
    --model_name EEMFlow \
    --input_type dt1 \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --wd $WD \
    --train_iters $TRAIN_ITERS \
    --val_iters $VAL_ITERS \
    --num_workers $NUM_WORKERS 2>&1 | tee training_output.log &

TRAIN_PID=$!
echo "Training started with PID: $TRAIN_PID"

# Wait for save directory to be created
sleep 15

# Find the save directory and move log there
SAVE_DIR=$(find /root/autodl-tmp/ssms_event_cameras/exp_HREM_meshflow/EEMFlow_dt1/ -name "bs${BATCH_SIZE}_*" -type d -newer /tmp 2>/dev/null | head -1)

if [ -n "$SAVE_DIR" ]; then
    echo "Save directory: $SAVE_DIR"
    mv training_output.log "$SAVE_DIR/console_output.log" 2>/dev/null
    echo "Console output will be saved to: $SAVE_DIR/console_output.log"
    echo ""
    echo "Monitor training with:"
    echo "  tail -f $SAVE_DIR/train.log          # Training metrics"
    echo "  tail -f $SAVE_DIR/console_output.log # Full console output"
    echo "  watch -n 2 nvidia-smi                 # GPU status"
else
    echo "Warning: Could not find save directory yet"
    echo "Console output in: ./training_output.log"
fi

echo ""
echo "Training is running in background (PID: $TRAIN_PID)"
echo "To stop training: pkill -f train_EEMFlow_HREM.py"
