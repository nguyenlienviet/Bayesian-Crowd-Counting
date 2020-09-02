#!/bin/sh
mkdir -p ./outdir

#DATADIR="/data/vnguy143/urban-forest/ml/data"
DATADIR="/data2/urban-forest"

python3 test.py \
    --image $DATADIR/SaMo_NAIP_60cm.tif \
    --tree_pts $DATADIR/SaMo_projected.csv \
    --data_split $DATADIR/SaMo_split.json \
    --save-dir ./outdir/$1 
