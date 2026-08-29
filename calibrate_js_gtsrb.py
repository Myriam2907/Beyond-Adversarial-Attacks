import json
import os
import argparse
import numpy as np

STOP_ID = 14
YIELD_ID = 13


def load_ours_mask(eval_dir, thresholds, weak_k=2):
    energy = np.load(os.path.join(eval_dir, "energy.npy"))
    confidence = np.load(os.path.join(eval_dir, "confidence.npy"))
    label = np.load(os.path.join(eval_dir, "label.npy"))
    conf2 = np.load(os.path.join(eval_dir, "2pass_conf_drop.npy"))
    l2_2 = np.load(os.path.join(eval_dir, "2pass_logit_l2.npy"))
    ch2 = np.load(os.path.join(eval_dir, "2pass_changed.npy"))
    conf3 = np.load(os.path.join(eval_dir, "3pass_max_conf_drop_true_stop_yield.npy"))
    l2_3 = np.load(os.path.join(eval_dir, "3pass_max_logit_l2_true_stop_yield.npy"))
    ch3 = np.load(os.path.join(eval_dir, "3pass_changed_true_stop_yield.npy"))

    n = len(energy)
    f_energy = energy > thresholds["energy_threshold"]
    f_conf = confidence < thresholds["confidence_min_threshold"]
    f_conf2 = conf2 > thresholds["conf_drop_2pass_threshold"]
    f_l2_2 = l2_2 > thresholds["logit_l2_2pass_threshold"]
    f_ch2 = ch2 == 1

    critical = (label == STOP_ID) | (label == YIELD_ID)
    f_conf3 = np.zeros(n, dtype=bool)
    f_l2_3 = np.zeros(n, dtype=bool)
    f_ch3 = np.zeros(n, dtype=bool)
    f_conf3[critical] = conf3[critical] > thresholds["conf_drop_3pass_threshold"]
    f_l2_3[critical] = l2_3[critical] > thresholds["logit_l2_3pass_threshold"]
    f_ch3[critical] = ch3[critical] == 1

    strong = f_ch2 | f_ch3 | f_l2_2
    weak_count = (
        f_energy.astype(int) + f_conf.astype(int) + f_conf2.astype(int)
        + f_conf3.astype(int) + f_l2_3.astype(int)
    )
    return strong | (weak_count >= int(weak_k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_eval", default="./eval_png_results_js/clean")
    ap.add_argument("--ours_thresholds", default="./anomaly_thresholds.json")
    ap.add_argument("--out", default="./js_thresholds_gtsrb.json")
    ap.add_argument("--weak_k", type=int, default=2)
    ap.add_argument("--js_fpr", type=float, default=0.05)
    ap.add_argument("--combined_fpr", type=float, default=0.06)
    args = ap.parse_args()

    with open(args.ours_thresholds, "r") as f:
        thresholds = json.load(f)

    js = np.load(os.path.join(args.clean_eval, "js.npy"))
    ours = load_ours_mask(args.clean_eval, thresholds, args.weak_k)
    n = len(js)

    js_only_threshold = float(np.quantile(js, 1.0 - args.js_fpr))
    js_only = js > js_only_threshold

    max_combined = int(np.floor(args.combined_fpr * n))
    budget = max_combined - int(ours.sum())
    not_ours_scores = js[~ours]

    if budget <= 0:
        combined_threshold = float(np.nextafter(js.max(), np.inf))
    elif budget >= len(not_ours_scores):
        combined_threshold = float(np.nextafter(js.min(), -np.inf))
    else:
        scores_desc = np.sort(not_ours_scores)[::-1]
        # Strict '>' threshold: choose the next score after the allowed additions.
        combined_threshold = float(scores_desc[budget])

    js_for_combined = js > combined_threshold
    combined = ours | js_for_combined

    out = {
        "js_transform": "224->208->224 bilinear, applied to normalized classifier input",
        "weak_k": int(args.weak_k),
        "n_clean": int(n),
        "ours_clean_fpr": float(ours.mean()),
        "js_only_target_fpr": float(args.js_fpr),
        "js_only_threshold": js_only_threshold,
        "js_only_actual_clean_fpr": float(js_only.mean()),
        "combined_target_fpr": float(args.combined_fpr),
        "combined_js_threshold": combined_threshold,
        "combined_actual_clean_fpr": float(combined.mean()),
        "combined_extra_js_flags": int((js_for_combined & ~ours).sum()),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("=" * 70)
    print("GTSRB JS CALIBRATION (NEW RESULTS ONLY)")
    print("=" * 70)
    print(f"OURS clean FPR:      {ours.mean()*100:.3f}%")
    print(f"JS clean FPR:        {js_only.mean()*100:.3f}%")
    print(f"OURS+JS clean FPR:   {combined.mean()*100:.3f}%")
    print(f"JS-only threshold:   {js_only_threshold:.10g}")
    print(f"Combined threshold:  {combined_threshold:.10g}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()