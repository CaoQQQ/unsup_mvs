#!/bin/bash

# dataset
DATASET_ROOT="/home/shu410/CQ/mvs_testing/cvp-dtu-test-1200/"
SELF_DATASET_ROOT="/home/shu410/CQ/mvs_testing/66/"
# checkpoint
LOAD_CKPT_DIR="/home/shu410/CQ/JDACS-MVS/jdacs-ms/checkpoints/jdacs-7/model_00000029.ckpt"
# logging
LOG_DIR="./logs/"
# output
OUT_DIR="./outputs/jdacs-7-66/"

CUDA_VISIBLE_DEVICES="0,1" python test.py \
	--info "jdacs-7-66/" \
	--mode "test" \
	--dataset_root $SELF_DATASET_ROOT \
	--imgsize 1200 \
	--nsrc 4 \
	--nscale 5 \
	--batch_size 1 \
	--loadckpt $LOAD_CKPT_DIR \
	--loggingdir $LOG_DIR \
	--outdir $OUT_DIR
