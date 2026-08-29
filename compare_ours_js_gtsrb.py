import argparse
import json
import os
import numpy as np

STOP_ID = 14
YIELD_ID = 13


def ours_mask(d, t, weak_k=2):
    energy = np.load(os.path.join(d, "energy.npy"))
    confidence = np.load(os.path.join(d, "confidence.npy"))
    label = np.load(os.path.join(d, "label.npy"))
    conf2 = np.load(os.path.join(d, "2pass_conf_drop.npy"))
    l2_2 = np.load(os.path.join(d, "2pass_logit_l2.npy"))
    ch2 = np.load(os.path.join(d, "2pass_changed.npy"))
    conf3 = np.load(os.path.join(d, "3pass_max_conf_drop_true_stop_yield.npy"))
    l2_3 = np.load(os.path.join(d, "3pass_max_logit_l2_true_stop_yield.npy"))
    ch3 = np.load(os.path.join(d, "3pass_changed_true_stop_yield.npy"))

    n = len(label)
    critical = (label == STOP_ID) | (label == YIELD_ID)
    f_conf3 = np.zeros(n, bool); f_l2_3 = np.zeros(n, bool); f_ch3 = np.zeros(n, bool)
    f_conf3[critical] = conf3[critical] > t["conf_drop_3pass_threshold"]
    f_l2_3[critical] = l2_3[critical] > t["logit_l2_3pass_threshold"]
    f_ch3[critical] = ch3[critical] == 1

    f_energy = energy > t["energy_threshold"]
    f_conf = confidence < t["confidence_min_threshold"]
    f_conf2 = conf2 > t["conf_drop_2pass_threshold"]
    f_l2_2 = l2_2 > t["logit_l2_2pass_threshold"]
    f_ch2 = ch2 == 1

    strong = f_ch2 | f_ch3 | f_l2_2
    weak = (f_energy.astype(int) + f_conf.astype(int) + f_conf2.astype(int)
            + f_conf3.astype(int) + f_l2_3.astype(int))
    return strong | (weak >= int(weak_k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", default="./eval_png_results_js")
    ap.add_argument("--ours_thresholds", default="./anomaly_thresholds.json")
    ap.add_argument("--js_thresholds", default="./js_thresholds_gtsrb.json")
    ap.add_argument("--attacks", default="clean,fgsm,patch,light")
    ap.add_argument("--out", default="./gtsrb_ours_js_results.json")
    args = ap.parse_args()

    with open(args.ours_thresholds) as f: t = json.load(f)
    with open(args.js_thresholds) as f: jt = json.load(f)
    weak_k = int(jt.get("weak_k", 2))
    js_t = float(jt["js_only_threshold"])
    combined_t = float(jt["combined_js_threshold"])

    results = {}
    print("=" * 100)
    print("GTSRB: OURS vs JS vs OURS+JS")
    print("=" * 100)
    print(f"{'Attack':<12}{'Accuracy':>11}{'OURS':>11}{'JS':>11}{'OURS+JS':>12}{'Wrong':>9}{'OURS wrong':>13}{'JS wrong':>11}{'+JS wrong':>12}")
    print("-" * 100)

    for attack in [x.strip() for x in args.attacks.split(',') if x.strip()]:
        d = os.path.join(args.eval_root, attack)
        pred = np.load(os.path.join(d, "pred.npy"))
        label = np.load(os.path.join(d, "label.npy"))
        js = np.load(os.path.join(d, "js.npy"))
        ours = ours_mask(d, t, weak_k)
        js_only = js > js_t
        combined = ours | (js > combined_t)
        wrong = pred != label
        wt = int(wrong.sum())

        def rate(mask): return float(mask.mean() * 100.0)
        def wrong_rate(mask): return float((mask & wrong).sum() / wt * 100.0) if wt else 0.0

        row = {
            "n": int(len(label)),
            "accuracy_percent": float((pred == label).mean() * 100.0),
            "ours_detection_percent": rate(ours),
            "js_detection_percent": rate(js_only),
            "ours_plus_js_detection_percent": rate(combined),
            "wrong_total": wt,
            "ours_wrong_caught": int((ours & wrong).sum()),
            "js_wrong_caught": int((js_only & wrong).sum()),
            "ours_plus_js_wrong_caught": int((combined & wrong).sum()),
            "ours_wrong_detection_percent": wrong_rate(ours),
            "js_wrong_detection_percent": wrong_rate(js_only),
            "ours_plus_js_wrong_detection_percent": wrong_rate(combined),
        }
        results[attack] = row
        print(f"{attack:<12}{row['accuracy_percent']:>10.2f}%{row['ours_detection_percent']:>10.2f}%{row['js_detection_percent']:>10.2f}%{row['ours_plus_js_detection_percent']:>11.2f}%{wt:>9}{row['ours_wrong_detection_percent']:>12.2f}%{row['js_wrong_detection_percent']:>10.2f}%{row['ours_plus_js_wrong_detection_percent']:>11.2f}%")

    with open(args.out, "w") as f:
        json.dump({"method": {"ours": "original tiered detector", "js": "224->208->224 JS", "ours_plus_js": "OURS OR calibrated JS"}, "results": results}, f, indent=2)
    print("-" * 100)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()