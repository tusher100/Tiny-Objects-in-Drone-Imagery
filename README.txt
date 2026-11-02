Supplementary material
Stable Label Assignment for Tiny Objects in Drone Imagery: An Entropic Optimal Transport Analysis

Contents
  ot_assignment_final_60epochs.ipynb   Kaggle notebook of the detection study (YOLOv8n, Ultralytics 8.3.0):
                                       data split, assigner patch, training and evaluation of the four runs,
                                       with its outputs (Section 5.4, Table 11, Figures 9 and 10).
  ot_assigner_src.py                   The entropic-OT assigner (Algorithm 1) as used in training.
  training_log.txt                     Complete console log of the four training runs; every detection
                                       number in the paper is read from this log.
  epoch_curves.json                    Per-epoch validation metrics and epoch times parsed from the log.
  make_paper_figures_notrain.py        Dataset statistics and the comparison of the two assigners on real
                                       VisDrone images at the first training iteration (Table 12, Figures 7, 8,
                                       11 and 12); runs on Kaggle without training.
  figs_notrain_stats.json              Output of the previous script.
  check_random_problems.py             Checks of Theorem 1 and of the regularization bias on random
                                       problems (Section 5.1).
  check_sinkhorn_rate.py               Check of the Sinkhorn rate (Section 5.1).
  check_row_error_and_overlaps.py      Row-marginal error and overlap statistics.
  check_assigner_scenes.py             Assigner-faithful scenes (Table 5, Figures 4 and 5).
  compare_assigners.py                 First assignment-level comparison, the Hilbert-metric check on random
                                       problems (Section 4.4) and a feasibility (Hall condition) check.
  compare_v2.py                        Tables 7-9: well-separated scenes with anchor-centred and object-aware
                                       predictions (Monte Carlo standard errors), crowded scenes, epsilon sweep,
                                       size-relative perturbations, selection margins vs changes (Sections 5.1-5.2).
  compare_v3.py                        Certified radius of Theorem 2(b), 50-iteration vs converged plan,
                                       projective diameter of a masked cost (Section 4.4).
  compare_v4.py                        Section 5.3: perturbation directions (Table 10), ground-truth annotation
                                       noise and the kappa sweep (Figure 6).
  make_beyond_figure.py                Figure 6 from compare_v4.json.
  *.json                               Outputs of the scripts above.
  make_figures.py                      Figures of the synthetic experiments.

Requirements: Python 3.10+, numpy, scipy, matplotlib; the notebook additionally needs ultralytics==8.3.0 and
pycocotools. The trained weights were not retained.
