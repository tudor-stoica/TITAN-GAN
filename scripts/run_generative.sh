#!/bin/bash
# Train TITAN-GAN and all baselines on the full patch set at one resolution.
# Usage: scripts/run_generative.sh 32|64
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
RES=${1:?usage: run_generative.sh 32|64}

if [ "$RES" = "32" ]; then
  OUT=./output; DIFF=./output/exp7_diffusion_v2; VAE_EPOCHS=500; DIFF_CKPT_EVERY=200
else
  OUT=./output/res64; DIFF=./output/diffusion_64; VAE_EPOCHS=250; DIFF_CKPT_EVERY=100
fi
DATA=$OUT/phase1_data
MASKS=$OUT/phase4_synthetic_data

best_epoch () {  # best_epoch <csv>  -> epoch with the highest SSIM
  $PY -c "import pandas as pd; d=pd.read_csv('$1'); print(int(d.loc[d['SSIM'].idxmax(),'Epoch']))"
}

echo "== Phase 1: patch extraction and Frangi + CCA vessel masks"
$PY phase1_generate_masks.py --num-img 0 --patch-size $RES --output-dir $DATA

echo "== Phase 2: Soft-Bridge"
$PY phase2_train_softbridge.py --data-dir $DATA --out-dir $OUT/models

echo "== Phase 3-4: VAE and synthetic masks"
$PY phase3_train_VAE.py --data-dir $DATA --out-dir $OUT/models --input-size $RES --epochs $VAE_EPOCHS
$PY phase4_gen_synth_masks.py --model-path $OUT/models/vae_geometry_heavy_best.pth --out-dir $MASKS --input-size $RES --num-samples 600

echo "== Phase 5-7: TITAN-GAN cGAN, checkpoint selection, synthetic CT"
$PY phase5_train_GAN.py --data-dir $DATA --phase2-path $OUT/models/soft_graph_bridge_best.pth --out-dir $OUT/models/tighterTitan
$PY phase6_model_selection.py --data-dir $DATA --models-dir $OUT/models/tighterTitan \
  --phase2-path $OUT/models/soft_graph_bridge_best.pth --out-dir $OUT/gan_evaluation
E=$(best_epoch $OUT/gan_evaluation/cgan_evaluation_metrics.csv)
$PY phase7_gen_synth_scans.py --mask-dir $MASKS --gen-path $(printf "$OUT/models/tighterTitan/cgan_gen_epoch_%03d.pth" $E) \
  --out-dir $OUT/phase7_final_dataset

echo "== Vanilla-TITAN (no topology loss)"
$PY exp2_vanilla_cgan.py --data-dir $DATA --out-dir $OUT/exp2_vanilla_cgan
$PY exp2_evaluate.py --data-dir $DATA --mask-dir $MASKS --models-dir $OUT/exp2_vanilla_cgan \
  --out-dir $OUT/exp2_vanilla_cgan/synthetic_data

echo "== WGAN-GP"
$PY exp1_train_gan.py --data-dir $DATA --out-dir $OUT/models/baseline_gan --patch-size $RES
$PY exp1_compare.py --data-dir $DATA --models-dir $OUT/models/baseline_gan --out-dir $OUT/exp1_baseline --patch-size $RES

echo "== alpha-WGAN"
$PY exp5_alpha_wgan.py --data-dir $DATA --out-dir $OUT/models/alpha_wgan --patch-size $RES
$PY exp5_compare.py --data-dir $DATA --models-dir $OUT/models/alpha_wgan --out-dir $OUT/exp5_alpha_wgan --patch-size $RES

echo "== Mask-conditioned diffusion"
$PY exp7_train_diffusion.py --data-dir $DATA --out-dir $DIFF --epochs 1400 --checkpoint-every $DIFF_CKPT_EVERY
$PY exp7_evaluate_diffusion.py --data-dir $DATA --models-dir $DIFF --out-dir $DIFF --num-final 1
E=$(best_epoch $DIFF/diffusion_evaluation.csv)
$PY mint_diffusion_lowdata.py --diffusion-ckpt $(printf "$DIFF/diffusion_unet_epoch_%03d.pth" $E) \
  --mask-dir $MASKS --out-dir $DIFF/synthetic_data

echo "== Done ($RES^3). Outputs in $OUT"
