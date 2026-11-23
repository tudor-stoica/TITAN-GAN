#!/bin/bash
# Evaluate the trained models at one resolution: topology (Betti-0/LCR), conditional
# mask-adherence, held-out evaluation (32^3) and inference timing.
# Run after scripts/run_generative.sh (and, for the held-out step, scripts/run_downstream.sh).
# The conditional-fidelity scripts load the best-epoch checkpoints listed at the top of
# each script; update those paths if model selection picked different epochs.
# Usage: scripts/run_evaluation.sh 32|64
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
RES=${1:?usage: run_evaluation.sh 32|64}
if [ "$RES" = "32" ]; then OUT=./output; DIFF=./output/exp7_diffusion_v2; else OUT=./output/res64; DIFF=./output/diffusion_64; fi

echo "== Topology (Frangi + CCA, Betti-0 / LCR), 100 samples per model"
$PY exp3_topology_eval.py --real-dir $OUT/phase1_data --baseline-dir $OUT/exp1_baseline/synthetic_data \
  --vanilla-dir $OUT/exp2_vanilla_cgan/synthetic_data --cascading-dir $OUT/phase7_final_dataset \
  --alpha-wgan-dir $OUT/exp5_alpha_wgan/synthetic_data --diffusion-dir $DIFF/synthetic_data \
  --num-samples 100 --out-dir $OUT/topology_results

echo "== Conditional mask-adherence"
if [ "$RES" = "32" ]; then
  E=$($PY -c "import pandas as pd; d=pd.read_csv('$DIFF/diffusion_evaluation.csv'); print(int(d.loc[d['SSIM'].idxmax(),'Epoch']))")
  $PY exp7_conditional_fidelity.py --diffusion-ckpt $(printf "$DIFF/diffusion_unet_epoch_%03d.pth" $E)
  $PY exp8_conditional_fidelity_vanilla_32.py
  if [ -d ./output/downstream/train_only/models/titan ]; then
    echo "== Held-out evaluation (train-split generators)"
    $PY exp12_heldout_eval_32.py
  fi
else
  $PY exp6_conditional_fidelity_diffusion_64.py
fi

echo "== Inference timing (all GAN-family models, both resolutions)"
$PY time_all_models.py
