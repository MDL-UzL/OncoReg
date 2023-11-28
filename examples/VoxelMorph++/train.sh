#!/bin/sh

datadir=$1
task=$2
outdir=$3

./train_vxmpp_supervised.py $datadir $task $outdir