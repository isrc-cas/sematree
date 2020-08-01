#!/bin/bash
uname -a
#date
#env
date
CS_PATH='./dataset/LIP'
LR=1e-3
WD=5e-4
BS=24
GPU_IDS=0
RESTORE_FROM='./dataset/resnet101-imagenet.pth'
INPUT_SIZE='384,384'
GIT_BRANCH=`git symbolic-ref --short -q HEAD`
SNAPSHOT_DIR='./snapshots/'$GIT_BRANCH
LOG_DIR='./logs/'$GIT_BRANCH
DATASET='train'
NUM_CLASSES=20
EPOCHS=150
DATE=`date +"%Y-%m-%d-%H-%M-%S"`

if [[ ! -e ${SNAPSHOT_DIR} ]]; then
    mkdir -p  ${SNAPSHOT_DIR}
fi

if [[ ! -e ${LOG_DIR} ]]; then
    mkdir -p  ${LOG_DIR}
fi

python train.py --data-dir ${CS_PATH} \
       --random-mirror\
       --random-scale\
       --restore-from ${RESTORE_FROM}\
       --gpu ${GPU_IDS}\
       --learning-rate ${LR}\
       --weight-decay ${WD}\
       --batch-size ${BS} \
       --input-size ${INPUT_SIZE}\
       --snapshot-dir ${SNAPSHOT_DIR}\
       --log-dir ${LOG_DIR}\
       --dataset ${DATASET}\
       --num-classes ${NUM_CLASSES} \
       --date ${DATE} \
       --epochs ${EPOCHS}

