# Notebooks

This directory now has three canonical notebooks for reproducing the paper workflow:

- `01_training_checkpoints.ipynb`
  Train the projector checkpoints used in the paper, including the mixed-teacher runs and the ablation variants.
- `02_evaluation_and_generalization.ipynb`
  Run the paper evaluation suite: in-domain checkpoint shootouts, cross-backend transfer, cross-dataset transfer, and Wasserstein/OOD analysis.
- `03_qps_and_result_audit.ipynb`
  Run the QPS experiments and audit the expected output files that back the paper tables and figures.

All generated artifacts should live under `notebooks/outputs/`, which is ignored by git.

The older `test_*.ipynb` files in this directory are exploratory history. They are not the recommended entry point for a reader trying to reproduce the paper.
