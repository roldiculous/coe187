#!/bin/sh
DEVICE="MAX78000"
TARGET="C:/MaximSDK/Examples/$DEVICE/CNN"
COMMON_ARGS="--device $DEVICE --timer 0 --display-checkpoint --verbose"

QUANTIZED_MODEL="C:/Users/Harold/Max78000/ai8x-training/logs/2025.11.09-215718/best-quantized.pth.tar"
YAML="C:\Users\Harold\Max78000\ai8x-synthesis\networks\rps-hwc.yaml"
SAMPLE="C:\Users\Harold\Max78000\ai8x-synthesis\tests\sample_rock_paper_scissors.npy"

python ai8xize.py --test-dir $TARGET --prefix rps_gen --overwrite --checkpoint-file $QUANTIZED_MODEL --config-file $YAML --sample-input $SAMPLE --fifo "$@" --softmax $COMMON_ARGS
