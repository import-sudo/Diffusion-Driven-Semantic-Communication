#!/bin/bash
#SBATCH -n 3
#SBATCH -w gpu06
#SBATCH --gres=gpu:1

# added by Miniconda3 4.5.12 installer
# >>> conda init >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$(CONDA_REPORT_ERRORS=false '/gpu01/miniconda3/bin/conda' shell.bash hook 2> /dev/null)"
if [ $? -eq 0 ]; then
    \eval "$__conda_setup"
else
    if [ -f "/gpu01/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/gpu01/miniconda3/etc/profile.d/conda.sh"
        CONDA_CHANGEPS1=false conda activate base
    else
        \export PATH="/gpu01/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda init <<<

#export CUDNN_PATH='/usr/local/cuda-9.0/lib64/libcudnn.so.7'
#export PATH=/usr/local/cuda-9.0/bin:$PATH
#export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH

conda activate ldm
date

cd $SLURM_SUBMIT_DIR

python scripts/train_submission.py --ckpt ../v2-1_512-ema-pruned.ckpt --config configs/stable-diffusion/v2-inference.yaml --init-img outputs/no_img_prompt/00000.png --strength 0.5 --ddim_step 400 --scale 0 --dataset church --n_samples 16 --save respond/mimo_v2_cddm/church/  --outdir respond/mimo_v2_cddm/church/   --lamda1 0.1 --lamda2 0.1  --lr 0.0001 --epochs 1 --CT mimo  --refine respond/mimo_v2_cddm/church/0_epoch_loss_3.389098985054236e-06.pth