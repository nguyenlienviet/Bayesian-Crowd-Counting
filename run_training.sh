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
    --batch-size 32 \
    --num-workers 1
