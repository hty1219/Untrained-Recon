#!/bin/bash
ulimit -s unlimited
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BART_TOOLBOX_PARALLEL=0
export PYTHONPATH=$PYTHONPATH:../

EXP_NAME="Gauss_Lips"
GPU_ID=7
DATA_PATH="./example_data" 
SAVE_PATH="./results"
DEPTH=2
WIDTH=64

CUDA_VISIBLE_DEVICES=$GPU_ID python mri.py \
    --task mri_knee \
    --folder_path "$DATA_PATH" \
    --exp_name "$EXP_NAME" \
    --model_type DIP_LPF \
    --num_scales $DEPTH \
    --dim $WIDTH \
    --upsample_mode nearest \
    --gaussian_blur_ks 5 \
    --Lipschitz_reg 1 \
    --lr 0.008 \
    --num_iters 3000 \
    --save_folder "$SAVE_PATH" \
    --verbose True

echo "Done! Results saved to $SAVE_PATH/$EXP_NAME"