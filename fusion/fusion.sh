#!/bin/bash

DTU_TEST_ROOT="/home/shu410/CQ/mvs_testing/cvp-dtu-test-1200/"
DEPTH_FOLDER="/home/shu410/CQ/JDACS-MVS/jdacs-ms/outputs/jdacs-3-test/"
FUSIBILE_EXE_PATH="/home/shu410/CQ/fusibile/build/fusibile"

CUDA_VISIBLE_DEVICES="0" python depthfusion.py \
        --dtu_test_root $DTU_TEST_ROOT \
        --depth_folder $DEPTH_FOLDER \
        --out_folder "fused_0.4_0.25" \
        --fusibile_exe_path $FUSIBILE_EXE_PATH \
        --prob_threshold 0.4 \
        --disp_threshold 0.25 \
        --num_consistent 3
