#!/bin/bash
DATASET_ROOT="/home/shu410/CQ/mvs_training/cvp-dtu-train-128/"

# Logging
CKPT_DIR="./outputs/"
LOG_DIR="./logs/"

python -m torch.distributed.launch --nproc_per_node=2 train.py \
	--info "jdacs-tf-version-1.0.1" \
	--log "nsrc-3-batch-8"\
	--mode "train" \
	--dataset_root $DATASET_ROOT \
	--imgsize  128 \
	--nsrc 3 \
	--nscale 2 \
	--epochs 30 \
	--lr 0.001 \
	--lrepochs "10,12,14,20:2" \
	--batch_size 8 \
	--loadckpt '' \
	--logckptdir $CKPT_DIR \
	--loggingdir $LOG_DIR \
	--resume 0  \
	--summarydir "summary" \
	--interval_scale 1.06 \
	--summary_freq 50 \
	--save_freq 1
