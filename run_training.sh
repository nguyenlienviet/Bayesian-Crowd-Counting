#!/bin/sh
mkdir -p ./outdir

DATADIR="/data/vnguy143/urban-forest/ml/data"

python3 train.py \
    --image $DATADIR/SaMo_NAIP_60cm.tif \
    --tree_pts $DATADIR/SaMo_projected.csv \
    --data_split $DATADIR/SaMo_split.json \
    --val-start 0 \
    --save-dir ./outdir \
    --crop-size 256 \
    --batch-size 1 \
    --num-workers 1 \
    --lr 1e-6 \
    --weight-decay 1e-4 \
    --downsample-ratio 8 \
    --sigma 8.0 \
    --background-ratio 0.15
