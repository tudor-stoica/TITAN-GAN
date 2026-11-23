#!/bin/bash
# Downstream nodule classification at 32^3 (full data and low-data sweep).
#  1. random and hard negatives
#  2. patient-level split (see make_original_split_hardneg.py)
#  3. every generator retrained on the training patients only, 150 synthetic positives each
#  4. classifiers: real-only + 4 augmentation conditions, 5 seeds
#  5. low-data sweep: N in {25,50,75}, seeds 42-46, all generators retrained per subset
# Requires output/phase1_data from scripts/run_generative.sh 32.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
S=./output/downstream/split_orig
SEEDS="42 43 44 45 46"

best_epoch () { $PY -c "import pandas as pd; d=pd.read_csv('$1'); print(int(d.loc[d['SSIM'].idxmax(),'Epoch']))"; }

# train_generators <subset_dir> <base_out_dir> <n_synth> <bs_bridge> <bs_vae> <bs_gan> <vae_epochs>
train_generators () {
  local SUB=$1 B=$2 NS=$3 BS2=$4 BS3=$5 BS5=$6 VE=$7
  local NE=$(ls $SUB/*_mask.npy | wc -l); [ $NE -gt 100 ] && NE=100
  $PY phase2_train_softbridge.py --data-dir $SUB --out-dir $B/models --batch-size $BS2
  $PY phase3_train_VAE.py --data-dir $SUB --out-dir $B/models --input-size 32 --epochs $VE --batch-size $BS3
  $PY phase4_gen_synth_masks.py --model-path $B/models/vae_geometry_heavy_best.pth --out-dir $B/phase4_synth --input-size 32 --num-samples $NS
  $PY phase5_train_GAN.py --data-dir $SUB --phase2-path $B/models/soft_graph_bridge_best.pth --out-dir $B/models/titan --batch-size $BS5
  $PY phase6_model_selection.py --data-dir $SUB --models-dir $B/models/titan --phase2-path $B/models/soft_graph_bridge_best.pth \
    --out-dir $B/titan_eval --num-samples $NE
  $PY phase7_gen_synth_scans.py --mask-dir $B/phase4_synth \
    --gen-path $(printf "$B/models/titan/cgan_gen_epoch_%03d.pth" $(best_epoch $B/titan_eval/cgan_evaluation_metrics.csv)) \
    --out-dir $B/synth_augment/titan
  $PY exp2_vanilla_cgan.py --data-dir $SUB --out-dir $B/vanilla --batch-size $BS5
  $PY exp2_evaluate.py --data-dir $SUB --mask-dir $B/phase4_synth --models-dir $B/vanilla --out-dir $B/synth_augment/vanilla --num-eval $NE
  $PY exp5_alpha_wgan.py --data-dir $SUB --out-dir $B/alpha_wgan --patch-size 32 --batch-size $BS5
  $PY exp5_compare.py --data-dir $SUB --models-dir $B/alpha_wgan --out-dir $B/alpha_wgan --patch-size 32 --num-eval $NE --num-final $NS
}

# classify <name> <real_train_pos_dir> <runs_dir> [synth_dir]
classify () {
  local extra=""; [ -n "${4:-}" ] && extra="--synth-pos-dir $4"
  $PY train_classifier.py --condition-name $1 --seed ${SEED:-42} \
    --real-train-pos-dir $2 --real-train-neg-dir $S/real_train_neg \
    --real-test-pos-dir $S/real_test_pos --real-test-neg-dir $S/real_test_neg \
    $extra --out-dir $3 --epochs 100 --batch-size 32
}

echo "== Negatives and patient-level split"
$PY phase8_extract_negatives.py
$PY phase8_extract_hard_negatives.py
$PY make_original_split_hardneg.py

echo "== Full data: generators on training patients only"
B=./output/downstream/train_only
train_generators $S/real_train_pos $B 150 32 64 8 250
$PY exp7_train_diffusion.py --data-dir $S/real_train_pos --out-dir $B/diffusion --epochs 1400 --checkpoint-every 200
$PY exp7_evaluate_diffusion.py --data-dir $S/real_train_pos --models-dir $B/diffusion --out-dir $B/diffusion --num-final 1
$PY mint_diffusion_lowdata.py --diffusion-ckpt $(printf "$B/diffusion/diffusion_unet_epoch_%03d.pth" $(best_epoch $B/diffusion/diffusion_evaluation.csv)) \
  --mask-dir $B/phase4_synth --out-dir $B/synth_augment/diffusion

echo "== Full data: classifiers"
R=./output/downstream/classifier_runs
for SEED in $SEEDS; do
  classify real_only_s$SEED $S/real_train_pos $R
  classify vanilla_s$SEED $S/real_train_pos $R $B/synth_augment/vanilla
  classify titan_s$SEED $S/real_train_pos $R $B/synth_augment/titan
  classify alpha_wgan_s$SEED $S/real_train_pos $R $B/alpha_wgan/synthetic_data
  classify diffusion_s$SEED $S/real_train_pos $R $B/synth_augment/diffusion
done
$PY summarize_classifier_runs.py fulldata --runs-dir $R

echo "== Low-data sweep"
R=./output/lowdata/classifier_runs
for N in 25 50 75; do
  if [ $N -eq 25 ]; then BSA="4 8 4"; else BSA="8 16 8"; fi
  for SEED in $SEEDS; do
    B=./output/lowdata/n${N}_s${SEED}
    $PY make_lowdata_subset.py --source-dir $S/real_train_pos --n $N --seed $SEED --out-dir $B/real_train_pos_subset
    train_generators $B/real_train_pos_subset $B $((N * 2)) $BSA 250
    classify real_only_n${N}_s${SEED} $B/real_train_pos_subset $R
    classify vanilla_n${N}_s${SEED} $B/real_train_pos_subset $R $B/synth_augment/vanilla
    classify titan_n${N}_s${SEED} $B/real_train_pos_subset $R $B/synth_augment/titan
    classify alpha_wgan_n${N}_s${SEED} $B/real_train_pos_subset $R $B/alpha_wgan/synthetic_data
  done
done
$PY summarize_classifier_runs.py lowdata --runs-dir $R
$PY evaluate_lowdata_curve.py --runs-dir $R --out-dir ./output/lowdata
