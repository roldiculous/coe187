#!/bin/sh
MODEL="ai85rpsnet"
DATASET="rock_paper_scissors"
QUANTIZED_MODEL="../ai8x-training/logs/2025.11.09-215718/best-quantized.pth.tar"

# evaluate scripts for cats vs dogs
python train.py --model $MODEL --dataset $DATASET --confusion --evaluate --exp-load-weights-from $QUANTIZED_MODEL  -8 --save-sample 1 --device MAX78000 "$@"

#evaluate scripts for kws
# python train.py --model $MODEL --dataset $DATASET --confusion --evaluate --exp-load-weights-from $QUANTIZED_MODEL -8 --save-sample 1 --device MAX78000 "$@"

