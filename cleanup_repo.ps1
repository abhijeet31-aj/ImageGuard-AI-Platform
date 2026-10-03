# ImageGuard repo cleanup — run this from the project ROOT folder
# (the folder that contains both python-engine/ and server/)
#
# This moves old exploratory scripts into archive/ folders (nothing
# is deleted that might have report value) and removes a few clearly
# pointless duplicate files. Verified beforehand: none of these files
# are imported by app.py or by the active Phase 4 pipeline scripts
# (phase4_common.py, extract_features.py, train_final_model.py,
# evaluate_final_model.py, error_analysis.py, evaluate_full_pipeline.py,
# verify_dataset.py, diagnose_prediction.py, synthesize_hard_manipulations.py).

# --------------------------------------------------
# 1. Archive old python-engine exploratory scripts
# --------------------------------------------------
New-Item -ItemType Directory -Force -Path "python-engine\archive" | Out-Null

$strayScripts = @(
    "analyze_ai_calibration.py",
    "analyze_edited_cases.py",
    "analyze_error_cases.py",
    "analyze_forensic_stats.py",
    "check_adm_shards.py",
    "evaluate_100_forensics.py",
    "evaluate_ai_model.py",
    "evaluate_external_adm.py",
    "evaluate_forensic_fusion.py",
    "evaluate_forensics.py",
    "evaluate_new_test.py",
    "external_adm_forensic_only.py",
    "external_adm_forensics.py",
    "external_adm_fusion.py",
    "external_adm_predictions.py",
    "extract_adm_1000.py",
    "fusion_error_analysis.py",
    "prepare_sample.py",
    "show_wrong_predictions.py",
    "test_ai.py",
    "threshold_analysis.py",
    "threshold_fusion_analysis.py"
)

foreach ($f in $strayScripts) {
    $path = "python-engine\$f"
    if (Test-Path $path) {
        git mv $path "python-engine\archive\$f"
    }
}

# --------------------------------------------------
# 2. Archive stray result CSVs (outputs of the scripts above)
# --------------------------------------------------
New-Item -ItemType Directory -Force -Path "python-engine\archive\old-results" | Out-Null

$strayResults = @(
    "new_test_predictions.csv",
    "ai_validation_predictions.csv",
    "external_adm_forensic_results.csv",
    "forensic_validation_results.csv",
    "external_adm_predictions.csv",
    "forensic_100_results.csv"
)

foreach ($f in $strayResults) {
    $path = "python-engine\$f"
    if (Test-Path $path) {
        git mv $path "python-engine\archive\old-results\$f"
    }
}

# --------------------------------------------------
# 3. Remove pointless duplicate CSV backups and stray log
# --------------------------------------------------
if (Test-Path "python-engine\(old)phase4_dataset_features.csv") {
    git rm "python-engine\(old)phase4_dataset_features.csv"
}
if (Test-Path "python-engine\(old2)phase4_dataset_features.csv") {
    git rm "python-engine\(old2)phase4_dataset_features.csv"
}
if (Test-Path "python-engine\note.md") {
    git rm "python-engine\note.md"
}

# --------------------------------------------------
# 4. Archive stray server test scripts
# --------------------------------------------------
New-Item -ItemType Directory -Force -Path "server\archive" | Out-Null

if (Test-Path "server\evaluate_20_images.js") {
    git mv "server\evaluate_20_images.js" "server\archive\evaluate_20_images.js"
}
if (Test-Path "server\test_20_images.js") {
    git mv "server\test_20_images.js" "server\archive\test_20_images.js"
}

# --------------------------------------------------
# 5. Commit
# --------------------------------------------------
git add -A
git commit -m "chore: archive old exploratory scripts and remove duplicate files"

Write-Host ""
Write-Host "Done. Review with 'git status' and 'git log -1 --stat', then push:"
Write-Host "    git push"