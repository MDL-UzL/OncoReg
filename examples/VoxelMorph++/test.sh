#!/bin/sh

datadir=$1
task=$2
mode=$3
model=$4
outdir=$5

./inference_vxmpp_OncoReg.py $datadir $task $mode $model $outdir