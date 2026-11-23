# TITAN-GAN

Code for **TITAN-GAN: A Tissue-Topology Aware Cascading Architecture for Fast Synthetic Computed Tomography Generation** (IEEE SSCI 2027).

TITAN-GAN generates 3D lung CT patches in two stages:

1. a 3D ResNet **VAE** generates a binary vessel mask (the "blueprint");
2. a 3D U-Net **conditional GAN** renders CT tissue over that mask.

During cGAN training, a small frozen CNN, the **Soft-Bridge**, is trained beforehand to imitate the Frangi + connected-component vessel extraction. It maps each generated CT back to a vessel map, and a Soft-Dice loss compares that map with the input mask.

The repository also contains the baselines used in the paper and the evaluation code:
- **Baselines:**
  - WGAN-GP
  - alpha-WGAN
  - Vanilla-TITAN, the same cGAN without the Soft-Bridge loss
  - a mask-conditioned 3D DDPM
- **Evaluation:**
  - SSIM/PSNR
  - Betti-0 / largest-component ratio
  - per-sample conditional mask adherence
  - timing
  - downstream nodule classification

## Requirements

- Python 3.10 and a CUDA GPU. The paper's experiments used an RTX 5060 Ti with 16 GB.
- Install the dependencies:

```bash
pip install -r requirements.txt
```

## Data

Download [LUNA16](https://luna16.grand-challenge.org/) and place it as follows. The paper used subsets 0-4.

```
data/luna16/
  annotations.csv
  candidates_V2/candidates_V2.csv
  subset0/ ... subset4/        # .mhd / .raw scans
```

Every script reads from `./data` and writes to `./output`, relative to the repository root.

## Quick start

```bash
# 1. Train TITAN-GAN and all baselines (32^3 and/or 64^3)
bash scripts/run_generative.sh 32
bash scripts/run_generative.sh 64

# 2. Downstream classification at 32^3: negatives, patient-level split,
#    train-only generators, full-data classifiers and the low-data sweep
bash scripts/run_downstream.sh

# 3. Evaluation: topology, conditional mask adherence, held-out evaluation, timing
bash scripts/run_evaluation.sh 32
bash scripts/run_evaluation.sh 64
```

Set `PY=/path/to/python` to use a specific interpreter.

At 32^3, step 1 takes a few hours on one GPU, mostly diffusion training. At 64^3 it takes well over a day.

## Scripts

| Stage | Script | Purpose |
|---|---|---|
| Data | `phase1_generate_masks.py` | Extract nodule-centred patches from LUNA16 and build Frangi + CCA vessel masks |
| TITAN-GAN | `phase2_train_softbridge.py` | Train the Soft-Bridge (CT → vessel map) |
| | `phase3_train_VAE.py`, `phase4_gen_synth_masks.py` | Train the mask VAE and sample synthetic blueprints |
| | `phase5_train_GAN.py` | Train the cGAN with adversarial, L1 and Soft-Bridge losses |
| | `phase6_model_selection.py`, `phase7_gen_synth_scans.py` | Pick the best checkpoint by SSIM and render synthetic CT |
| Baselines | `exp1_train_gan.py`, `exp1_compare.py` | WGAN-GP |
| | `exp2_vanilla_cgan.py`, `exp2_evaluate.py` | Vanilla-TITAN (no topology loss) |
| | `exp5_alpha_wgan.py`, `exp5_compare.py` | alpha-WGAN |
| | `exp7_train_diffusion.py`, `exp7_evaluate_diffusion.py`, `mint_diffusion_lowdata.py`, `mint_diffusion_64.py` | Mask-conditioned 3D DDPM (DDIM sampling) |
| Evaluation | `exp3_topology_eval.py` | Betti-0 and LCR of real and generated patches |
| | `exp7_conditional_fidelity.py`, `exp8_conditional_fidelity_vanilla_32.py`, `exp6_conditional_fidelity_diffusion_64.py` | Per-sample conditional mask adherence with paired Wilcoxon tests |
| | `exp12_heldout_eval_32.py` | Held-out SSIM/PSNR, mask adherence and Soft-Bridge check |
| | `time_all_models.py` | Inference timing |
| | `exp4_visuals.py` | Qualitative comparison figure |
| Downstream | `phase8_extract_negatives.py`, `phase8_extract_hard_negatives.py` | Random and vesselness-ranked (hard) negatives |
| | `make_downstream_split.py`, `make_original_split_hardneg.py` | Patient-level train/test split |
| | `classifier_3d.py`, `train_classifier.py` | 3D CNN nodule classifier |
| | `make_lowdata_subset.py`, `summarize_classifier_runs.py`, `evaluate_lowdata_curve.py` | Low-data sweep and summaries |

Every script has `--help`.

## Notes

- **Checkpoint selection.** Each generator's checkpoint is chosen by SSIM on training data:
  - TITAN-GAN uses `phase6_model_selection.py`;
  - Vanilla-TITAN uses `exp2_evaluate.py`;
  - WGAN-GP and alpha-WGAN use `exp1_compare.py` and `exp5_compare.py`;
  - the diffusion model uses `exp7_evaluate_diffusion.py`.

  The conditional-fidelity scripts, `time_all_models.py` and `exp4_visuals.py` load fixed checkpoint paths (the epochs selected in the paper). If you retrain, update those paths to the epochs your selection picked.
- **Unconditional baselines.** WGAN-GP and alpha-WGAN have no paired target, so their SSIM/PSNR compares each sample with an arbitrary real patch. It is not comparable with the conditioned models.
- **Downstream split.** All generators used for augmentation are trained only on the training patients of the downstream split. The split is patient-level: no scan appears in both train and test.

## Citation

```bibtex
@inproceedings{stoica2027titangan,
  title     = {{TITAN-GAN}: A Tissue-Topology Aware Cascading Architecture for Fast Synthetic Computed Tomography Generation},
  author    = {Stoica, Tudor and Gan, Junxian and Liu, Yang and Asilian Bidgoli, Azam},
  booktitle = {IEEE Symposium Series on Computational Intelligence (SSCI)},
  year      = {2027}
}
```
