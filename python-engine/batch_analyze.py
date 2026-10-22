"""
batch_analyze.py  --  ImageGuard pipeline ko ek folder ki SAARI images par chalata hai.

Ye bilkul wahi steps same order mein chalata hai jo app.py ka /analyze-image
endpoint chalata hai (Community Forensics ViT, coarse regional ViT, metadata,
ELA, forgery, manipulation, binary fusion, final meta-fusion), bas FastAPI ke bina.

READ-ONLY: ye
  - koi model/weights overwrite nahi karta,
  - ela_output.jpg nahi likhta,
  - utils.csv_logger / forgery_csv_logger ko call NAHI karta
    (taaki aapki training logs/CSV mein test images ki rows na judein).
Sirf batch_output/ folder mein results likhta hai.

Kaise chalana hai (python-engine folder ke andar, venv active ke saath):
    python batch_analyze.py                      # test_images/ ki saari images
    python batch_analyze.py path\\to\\folder      # koi aur folder
    python batch_analyze.py --deep               # + Deep Scan (tiled ViT, slow)
    python batch_analyze.py --recursive          # subfolders bhi

Output:
    batch_output/batch_results.csv    (flat table, ek image = ek row)
    batch_output/batch_results.json   (har image ka poora raw response)
"""
import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
CONTENT_TYPES = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".webp": "image/webp", ".bmp": "image/bmp",
    ".tif": "image/tiff", ".tiff": "image/tiff",
}


def load_pipeline():
    """Import everything the real pipeline uses (ViT loads on import of ai.detector)."""
    print("Loading pipeline (ViT model load hone mein kuch der lag sakti hai)...")
    from cv.image_reader import read_image
    from cv.validator import validate_image
    from cv.metadata import extract_metadata, analyze_metadata
    from cv.ela import generate_difference_image, calculate_ela_metrics
    from cv.forgery import analyze_forgery
    from cv.manipulation import analyze_manipulation
    from ai.detector import detect_ai_image, detect_ai_tiled, detect_ai_coarse_regions
    from services.fusion import predict_fusion
    import services.final_fusion as final_fusion_module
    from services.final_fusion import combine_final_analysis, refine_with_tile_analysis

    return dict(
        read_image=read_image, validate_image=validate_image,
        extract_metadata=extract_metadata, analyze_metadata=analyze_metadata,
        generate_difference_image=generate_difference_image,
        calculate_ela_metrics=calculate_ela_metrics,
        analyze_forgery=analyze_forgery, analyze_manipulation=analyze_manipulation,
        detect_ai_image=detect_ai_image, detect_ai_tiled=detect_ai_tiled,
        detect_ai_coarse_regions=detect_ai_coarse_regions,
        predict_fusion=predict_fusion,
        combine_final_analysis=combine_final_analysis,
        refine_with_tile_analysis=refine_with_tile_analysis,
        final_fusion_module=final_fusion_module,
    )


def class_probabilities(P, fusion_analysis, manipulation_analysis):
    """Re-compute the trained meta-model's 4 class probabilities + OOD score for display.
    Uses the same private helpers/feature order as services/final_fusion.py.
    Returns ({} , None) if no trained model is loaded or anything differs."""
    try:
        import numpy as np
        ff = P["final_fusion_module"]
        model = getattr(ff, "_TRAINED_MODEL", None)
        if model is None:
            return {}, None
        forensic = fusion_analysis.get("forensicFeatures", {})
        vec = [
            ff._probability_to_log_odds(float(fusion_analysis.get("ai_probability", 0.0))),
            ff._probability_to_log_odds(float(manipulation_analysis.get("manipulationProbability", 0.0))),
            float(fusion_analysis.get("ai_regional_std") or 0.0),
            *[float(forensic.get(n, 0.0)) for n in ff._FORENSIC_FEATURE_NAMES],
        ]
        x = np.array(vec, dtype=np.float64)
        z = (x - np.array(model["mean"])) / np.array(model["std"])
        probs = ff._softmax(z @ np.array(model["weights"]) + np.array(model["bias"]))
        out = {c: round(float(p), 4) for c, p in zip(model["class_names"], probs.tolist())}
        return out, round(float(np.sqrt(np.mean(z ** 2))), 3)
    except Exception:
        return {}, None


def analyze_one(P, path, deep=False):
    """Same order as app.py /analyze-image."""
    image_bytes = path.read_bytes()
    content_type = CONTENT_TYPES.get(path.suffix.lower(), "image/jpeg")

    success, image_cv, image_info, error = P["read_image"](image_bytes)
    if not success:
        return {"success": False, "message": error}

    valid, message = P["validate_image"](image_cv, image_info, content_type)
    if not valid:
        return {"success": False, "message": message}

    # AI detection needs a file path -> use the original file directly (read-only)
    ai_detection = P["detect_ai_image"](str(path))
    coarse_regions = P["detect_ai_coarse_regions"](str(path))

    metadata = P["extract_metadata"](image_bytes)
    metadata_analysis = P["analyze_metadata"](metadata)

    difference_image = P["generate_difference_image"](image_bytes)
    ela_metrics = P["calculate_ela_metrics"](difference_image)

    forgery_analysis = P["analyze_forgery"](image_cv)
    manipulation_analysis = P["analyze_manipulation"](image_cv, difference_image)

    fusion_analysis = P["predict_fusion"](
        ai_probability=ai_detection["ai_probability"],
        forensic_analysis={
            "meanDifference": ela_metrics["meanDifference"],
            "maxDifference": ela_metrics["maxDifference"],
            "standardDeviation": ela_metrics["standardDeviation"],
            "edgeDensity": forgery_analysis["edgeDensity"],
            "noiseMean": forgery_analysis["noiseMean"],
            "noiseStd": forgery_analysis["noiseStd"],
            "blurScore": forgery_analysis["blurScore"],
            "sharpnessScore": forgery_analysis["sharpnessScore"],
            "blockSharpnessAverage": forgery_analysis["blockSharpnessAverage"],
            "blockSharpnessStd": forgery_analysis["blockSharpnessStd"],
        },
        ai_regional_std=coarse_regions["regionalStd"],
        metadata_integrity_score=metadata_analysis["integrityScore"],
    )

    final_analysis = P["combine_final_analysis"](fusion_analysis, manipulation_analysis)
    probs, ood_score = class_probabilities(P, fusion_analysis, manipulation_analysis)

    result = {
        "success": True,
        "filename": path.name,
        "contentType": content_type,
        **image_info,
        "aiDetection": ai_detection,
        "coarseRegionAnalysis": coarse_regions,
        "metadata": metadata,
        "metadataAnalysis": metadata_analysis,
        "elaMetrics": ela_metrics,
        "forgeryAnalysis": forgery_analysis,
        "manipulationAnalysis": manipulation_analysis,
        "fusionAnalysis": fusion_analysis,
        "finalAnalysis": final_analysis,
        "_classProbabilities": probs,
        "_oodScore": ood_score,
    }

    if deep:
        tile = P["detect_ai_tiled"](str(path))
        result["tileAnalysis"] = tile
        result["refinedFinalAnalysis"] = P["refine_with_tile_analysis"](final_analysis, tile)

    return result


def flatten(r):
    """One flat CSV row from the full response."""
    if not r.get("success"):
        return {"filename": r.get("filename", ""), "error": r.get("message", "failed")}
    ai = r["aiDetection"]; co = r["coarseRegionAnalysis"]; ma = r["manipulationAnalysis"]
    el = r["elaMetrics"]; fo = r["forgeryAnalysis"]; fu = r["fusionAnalysis"]
    fi = r["finalAnalysis"]; me = r["metadataAnalysis"]; pr = r.get("_classProbabilities", {})
    row = {
        "filename": r["filename"],
        "width": r.get("width"), "height": r.get("height"),
        "megapixels": round((r.get("width", 0) * r.get("height", 0)) / 1e6, 2),
        "ai_prediction": ai.get("prediction"), "ai_probability": ai.get("ai_probability"),
        "ai_confidence": ai.get("confidence"),
        "ai_regional_std": co.get("regionalStd"),
        "metadata_integrity": me.get("integrityScore"), "metadata_risk": me.get("riskLevel"),
        "ela_mean": el.get("meanDifference"), "ela_max": el.get("maxDifference"),
        "ela_std": el.get("standardDeviation"),
        "edgeDensity": fo.get("edgeDensity"), "noiseMean": fo.get("noiseMean"),
        "noiseStd": fo.get("noiseStd"), "blurScore": fo.get("blurScore"),
        "sharpnessScore": fo.get("sharpnessScore"),
        "blockSharpnessAverage": fo.get("blockSharpnessAverage"),
        "blockSharpnessStd": fo.get("blockSharpnessStd"),
        "manip_prediction": ma.get("prediction"),
        "manip_probability": ma.get("manipulationProbability"),
        "manip_confidence": ma.get("confidence"),
        "manip_evidence": " | ".join(ma.get("evidence", []) or []),
        "fusion_prediction": fu.get("prediction"),
        "fusion_probability": fu.get("fusion_probability"),
        "FINAL_prediction": fi.get("prediction"),
        "final_confidence": fi.get("confidence"),
        "final_reliability": fi.get("reliability"),
        "p_Authentic": pr.get("Authentic"), "p_AI_Generated": pr.get("AI Generated"),
        "p_Manipulated": pr.get("Manipulated"), "p_Partially_AI_Edited": pr.get("Partially AI-Edited"),
        "ood_rms_z": r.get("_oodScore"),
        "final_evidence": " | ".join(fi.get("evidence", []) or []),
    }
    if "tileAnalysis" in r:
        t = r["tileAnalysis"]
        row["deep_tileVerdict"] = t.get("tileVerdict")
        row["deep_suspicious_regions"] = len(t.get("suspiciousRegions", []) or [])
        row["deep_refined_prediction"] = (r.get("refinedFinalAnalysis") or {}).get("prediction")
    return row


def main():
    ap = argparse.ArgumentParser(description="Batch ImageGuard analysis (with ViT).")
    ap.add_argument("folder", nargs="?", default=str(HERE / "test_images"))
    ap.add_argument("--deep", action="store_true", help="also run Deep Scan (slow)")
    ap.add_argument("--recursive", action="store_true", help="include subfolders")
    ap.add_argument("--out", default=str(HERE / "batch_output"))
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        print(f"ERROR: folder nahi mila: {folder}")
        sys.exit(1)

    pattern = "**/*" if args.recursive else "*"
    images = sorted(p for p in folder.glob(pattern) if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"ERROR: {folder} mein koi image nahi mili.")
        sys.exit(1)

    P = load_pipeline()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{len(images)} images analyse hongi {'(Deep Scan ON)' if args.deep else ''}\n")
    full, rows = [], []
    for i, path in enumerate(images, 1):
        t0 = time.time()
        try:
            r = analyze_one(P, path, deep=args.deep)
        except Exception as e:  # noqa: BLE001  - ek image fail ho to baaki chalti rahein
            traceback.print_exc()
            r = {"success": False, "filename": path.name, "message": f"{type(e).__name__}: {e}"}
        r["_seconds"] = round(time.time() - t0, 1)
        full.append(r)
        row = flatten(r)
        row["seconds"] = r["_seconds"]
        rows.append(row)
        if r.get("success"):
            print(f"[{i:>2}/{len(images)}] {path.name:<22} AI={row['ai_probability']:<7} "
                  f"manip={row['manip_probability']:<7} -> {row['FINAL_prediction']:<20} "
                  f"(P:Auth={row['p_Authentic']} AI={row['p_AI_Generated']} "
                  f"Man={row['p_Manipulated']} Part={row['p_Partially_AI_Edited']}) "
                  f"[{r['_seconds']}s]")
        else:
            print(f"[{i:>2}/{len(images)}] {path.name:<22} FAILED: {r.get('message')}")

    # ---- save outputs ----
    cols = []
    for row in rows:
        for k in row:
            if k not in cols:
                cols.append(k)
    csv_path = out_dir / "batch_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    json_path = out_dir / "batch_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full, f, indent=2, default=str)

    print(f"\nDone. Saved:\n  {csv_path}\n  {json_path}")


if __name__ == "__main__":
    main()