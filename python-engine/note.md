======================================================================
IMAGEGUARD — PHASE 4 META-FUSION MODEL TRAINING
======================================================================

Total labelled images available: 6000

TRAIN — 4200 images
    Authentic              1050
    AI Generated           1050
    Manipulated            1050
    Partially AI-Edited    1050

VAL — 900 images
    Authentic              225
    AI Generated           225
    Manipulated            225
    Partially AI-Edited    225

TEST — 900 images
    Authentic              225
    AI Generated           225
    Manipulated            225
    Partially AI-Edited    225

Cross-validating L2 over 6 candidate(s), 5 folds each (val/test sets are NOT used for this):
  L2=0.001    avg macro-F1 across folds = 0.8103
  L2=0.01     avg macro-F1 across folds = 0.8103
  L2=0.05     avg macro-F1 across folds = 0.8101
  L2=0.1      avg macro-F1 across folds = 0.8101
  L2=0.3      avg macro-F1 across folds = 0.8099
  L2=1.0      avg macro-F1 across folds = 0.8099

Selected L2=0.001 (best cross-validated macro-F1: 0.8103)

Training softmax regression (13 features, 4 classes, 6000 iterations)...

Train accuracy: 81.45%
Val accuracy:   79.89%
Test accuracy:  80.56%

(Train accuracy is NOT the model's real-world accuracy — run evaluate_final_model.py for the honest per-class precision/recall/F1/confusion matrix on the held-out test set.)

Reliability thresholds (calibrated from validation data):
  Severe (-> Needs Review)  : confidence < 0.454 or margin < 0.052
  Medium (-> shown, flagged): confidence < 0.583 or margin < 0.262
  -> 51/900 (5.7%) would be Needs Review
  -> 108/900 (12.0%) would be shown with Medium reliability
  -> 741/900 would be shown with High reliability

Cross-check thresholds (calibrated from validation data):
  Manipulated cross-check threshold: 0.078 (was a fixed 0.35 guess before)
  AI Generated cross-check threshold: 0.197 (was a fixed 0.35 guess before)

Out-of-distribution threshold (97.5th percentile of validation RMS z-scores): 2.289
Severe OOD threshold (99.5th percentile): 3.549
Live images whose overall feature profile is farther from the training distribution than this will be reported as 'Needs Review' regardless of predicted class (see final_fusion.py).

Model saved to: D:\Users\ABHIJEET\Downloads\7th-sem-project\ImageGuard\python-engine\models\final_fusion_model.json
======================================================================
(venv) PS D:\Users\ABHIJEET\Downloads\7th-sem-project\ImageGuard\python-engine>  python evaluate_final_model.py
======================================================================
IMAGEGUARD — PHASE 4 MODEL EVALUATION (held-out test set)
======================================================================

TEST (held out, never seen during training) — 900 images
    Authentic              225
    AI Generated           225
    Manipulated            225
    Partially AI-Edited    225

Test accuracy: 80.56%

Per-class metrics:
Class                    Support   Precision    Recall      F1
Authentic                    225      68.68%    85.78%  76.28%
AI Generated                 225      98.64%    96.89%  97.76%
Manipulated                  225      82.38%    70.67%  76.08%
Partially AI-Edited          225      75.61%    68.89%  72.09%

Macro avg — Precision: 81.33%  Recall: 80.56%  F1: 80.55%

Out-of-distribution gate (threshold=2.289):
  Flagged as Needs Review (OOD): 22 / 900 (2.4%)
  Accuracy on the remaining (in-distribution) images: 80.30%
  Accuracy on the flagged (out-of-distribution) images: 90.91%
  (low accuracy here is EXPECTED and is exactly why these get flagged)

Confusion matrix (rows = true class, columns = predicted class):
                             Authentic  AI Generated   Manipulated  Partially AI
Authentic                          193             0            13            19
AI Generated                         1           218             2             4
Manipulated                         39             0           159            27
Partially AI-Edited                 48             3            19           155

Full report saved to: D:\Users\ABHIJEET\Downloads\7th-sem-project\ImageGuard\python-engine\phase4_evaluation_report.csv
======================================================================
(venv) PS D:\Users\ABHIJEET\Downloads\^Ch-sem-project\ImageGuard\python-engine> 
(venv) PS D:\Users\ABHIJEET\Downloads\7th-sem-project\ImageGuard\python-engine>  python evaluate_full_pipeline.py
==============================================================================
IMAGEGUARD — FULL PIPELINE EVALUATION (all safety layers, held-out test set)
==============================================================================

Test set: 900 images

Needs Review triggered: 98 / 900 (10.9%)
Answered (High + Medium reliability): 802 / 900 (89.1%)
Accuracy on all answered predictions: 83.04%

Reliability tier breakdown (should show High > Medium in accuracy):
  High      683 images (75.9% of all)  accuracy: 87.41%
  Medium    119 images (13.2% of all)  accuracy: 57.98%

Why Needs Review triggered (by safety layer):
  Low confidence / narrow margin        73  (8.1% of all test images)
  Cross-check conflict                  23  (2.6% of all test images)
  OOD gate                               2  (0.2% of all test images)

Needs Review rate BY TRUE CLASS (which classes get over-flagged):
  Authentic              22/225 (9.8%) flagged Needs Review
  AI Generated           22/225 (9.8%) flagged Needs Review
  Manipulated            29/225 (12.9%) flagged Needs Review
  Partially AI-Edited    25/225 (11.1%) flagged Needs Review

==============================================================================
If 'Low confidence / narrow margin' dominates, the general margin/confidence thresholds are too strict for this model's actual probability spread — they were guessed, not measured, and should be relaxed based on the numbers above.
==============================================================================
(venv) PS D:\Users\ABHIJEET\Downloads\7th-sem-project\ImageGuard\python-engine>     