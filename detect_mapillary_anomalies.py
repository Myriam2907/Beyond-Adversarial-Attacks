import os
import json
import argparse
import numpy as np


DEFAULT_ATTACKS = [
    "clean",
    "fgsm",
    "rfgsm",
    "pgd",
    "patch",
    "gaussian",
    "salt_pepper",
    "light",
    "fog",
    "motion_blur",
]

CRITICAL_IDS = [241, 242, 243, 265]


def load_thresholds(path):
    with open(path, "r") as f:
        return json.load(f)


def load_array(eval_dir, name, required=True):
    path = os.path.join(eval_dir, name)
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"Missing required file: {path}")
        return None
    return np.load(path, allow_pickle=True)


def detect_one(eval_dir, thresholds, attack_name, weak_k=3):
    print("\n" + "=" * 70)
    print(f"DETECT: {attack_name}")
    print("=" * 70)

    energy = load_array(eval_dir, "energy.npy")
    confidence = load_array(eval_dir, "confidence.npy")
    pred = load_array(eval_dir, "pred.npy")
    label = load_array(eval_dir, "label.npy")

    conf_drop_2 = load_array(eval_dir, "2pass_conf_drop.npy")
    logit_l2_2 = load_array(eval_dir, "2pass_logit_l2.npy")
    changed_2 = load_array(eval_dir, "2pass_changed.npy")

    conf_drop_3 = load_array(eval_dir, "3pass_max_conf_drop_critical.npy", required=False)
    logit_l2_3 = load_array(eval_dir, "3pass_max_logit_l2_critical.npy", required=False)
    changed_3 = load_array(eval_dir, "3pass_changed_critical.npy", required=False)

    n = len(label)

    if conf_drop_3 is None:
        conf_drop_3 = np.full(n, -1.0)
    if logit_l2_3 is None:
        logit_l2_3 = np.full(n, -1.0)
    if changed_3 is None:
        changed_3 = np.full(n, -1)

    flag_energy = energy > thresholds["energy_threshold"]
    flag_confidence = confidence < thresholds["confidence_min_threshold"]
    flag_conf_drop_2 = conf_drop_2 > thresholds["conf_drop_2pass_threshold"]
    flag_logit_l2_2 = logit_l2_2 > thresholds["logit_l2_2pass_threshold"]
    flag_changed_2 = changed_2 == 1

    is_critical = np.isin(label, CRITICAL_IDS)
    valid_3 = is_critical & (changed_3 != -1)

    flag_conf_drop_3 = np.zeros(n, dtype=bool)
    flag_logit_l2_3 = np.zeros(n, dtype=bool)
    flag_changed_3 = np.zeros(n, dtype=bool)

    if "conf_drop_3pass_threshold" in thresholds:
        flag_conf_drop_3[valid_3] = (
            conf_drop_3[valid_3] > thresholds["conf_drop_3pass_threshold"]
        )

    if "logit_l2_3pass_threshold" in thresholds:
        flag_logit_l2_3[valid_3] = (
            logit_l2_3[valid_3] > thresholds["logit_l2_3pass_threshold"]
        )

    flag_changed_3[valid_3] = changed_3[valid_3] == 1

    strong_signals = (
        flag_changed_2
        | flag_changed_3
        | flag_logit_l2_2
    )

    weak_count = (
        flag_energy.astype(int)
        + flag_confidence.astype(int)
        + flag_conf_drop_2.astype(int)
        + flag_conf_drop_3.astype(int)
        + flag_logit_l2_3.astype(int)
    )

    suspicious = strong_signals | (weak_count >= weak_k)

    correct = pred == label
    wrong = ~correct

    n_suspicious = int(suspicious.sum())
    suspicious_rate = n_suspicious / n * 100.0

    accuracy = correct.mean() * 100.0

    wrong_total = int(wrong.sum())
    wrong_flagged = int((suspicious & wrong).sum())
    wrong_not_flagged = int((~suspicious & wrong).sum())

    detection_rate_wrong = (
        wrong_flagged / wrong_total * 100.0
        if wrong_total > 0 else 0.0
    )

    is_clean = attack_name == "clean"

    flag_count = (
        flag_energy.astype(int)
        + flag_confidence.astype(int)
        + flag_conf_drop_2.astype(int)
        + flag_logit_l2_2.astype(int)
        + flag_changed_2.astype(int)
        + flag_conf_drop_3.astype(int)
        + flag_logit_l2_3.astype(int)
        + flag_changed_3.astype(int)
    )

    results = {
        "attack_name": attack_name,
        "n_samples": int(n),
        "accuracy_percent": float(accuracy),

        "n_suspicious": n_suspicious,
        "suspicious_rate_percent": float(suspicious_rate),

        "fpr_clean_percent": float(suspicious_rate) if is_clean else None,
        "tpr_attack_percent": None if is_clean else float(suspicious_rate),

        "wrong_total": wrong_total,
        "wrong_flagged": wrong_flagged,
        "wrong_not_flagged": wrong_not_flagged,
        "detection_rate_of_wrong_percent": float(detection_rate_wrong),

        "weak_k": int(weak_k),
        "critical_ids_used": CRITICAL_IDS,
        "critical_samples": int(is_critical.sum()),
        "valid_3pass_samples": int(valid_3.sum()),

        "flags": {
            "energy": int(flag_energy.sum()),
            "confidence": int(flag_confidence.sum()),
            "conf_drop_2pass": int(flag_conf_drop_2.sum()),
            "logit_l2_2pass": int(flag_logit_l2_2.sum()),
            "changed_2pass": int(flag_changed_2.sum()),
            "conf_drop_3pass_critical": int(flag_conf_drop_3.sum()),
            "logit_l2_3pass_critical": int(flag_logit_l2_3.sum()),
            "changed_3pass_critical": int(flag_changed_3.sum()),
        },

        "tiered_breakdown": {
            "strong_only": int((strong_signals & ~(weak_count >= weak_k)).sum()),
            "weak_only": int((~strong_signals & (weak_count >= weak_k)).sum()),
            "both": int((strong_signals & (weak_count >= weak_k)).sum()),
        },

        "flag_statistics": {
            "avg_flags_per_sample": float(flag_count.mean()),
            "samples_with_0_flags": int((flag_count == 0).sum()),
            "samples_with_1_flag": int((flag_count == 1).sum()),
            "samples_with_2_flags": int((flag_count == 2).sum()),
            "samples_with_3_flags": int((flag_count == 3).sum()),
            "samples_with_4plus_flags": int((flag_count >= 4).sum()),
        },
    }

    out_json = os.path.join(eval_dir, "anomaly_detection_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    suspicious_idx = np.where(suspicious)[0]
    np.save(os.path.join(eval_dir, "suspicious_indices.npy"), suspicious_idx)

    if os.path.exists(os.path.join(eval_dir, "filenames.npy")):
        filenames = np.load(os.path.join(eval_dir, "filenames.npy"), allow_pickle=True)
        out_txt = os.path.join(eval_dir, "suspicious_files.txt")
        with open(out_txt, "w") as f:
            for p in filenames[suspicious_idx]:
                f.write(str(p) + "\n")

    print(f"Accuracy: {accuracy:.2f}%")
    if is_clean:
        print(f"FPR clean: {suspicious_rate:.2f}%")
    else:
        print(f"TPR attack: {suspicious_rate:.2f}%")
    print(f"Wrong predictions: {wrong_total}")
    print(f"Wrong flagged: {wrong_flagged}")
    print(f"Wrong missed: {wrong_not_flagged}")
    print(f"Detection rate of wrong: {detection_rate_wrong:.2f}%")
    print(f"Valid 3-pass samples: {int(valid_3.sum())}")
    print("Saved:", out_json)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_root", default="./mapillary_eval_results")
    parser.add_argument(
        "--thresholds",
        default="./mapillary_thresholds_clean/mapillary_anomaly_thresholds.json"
    )
    parser.add_argument("--attacks", nargs="+", default=DEFAULT_ATTACKS)
    parser.add_argument("--weak_k", type=int, default=3)
    args = parser.parse_args()

    thresholds = load_thresholds(args.thresholds)

    all_results = {}

    for attack in args.attacks:
        eval_dir = os.path.join(args.eval_root, attack)

        if not os.path.isdir(eval_dir):
            print("Missing eval dir, skipping:", eval_dir)
            continue

        all_results[attack] = detect_one(
            eval_dir=eval_dir,
            thresholds=thresholds,
            attack_name=attack,
            weak_k=args.weak_k,
        )

    out_all = os.path.join(args.eval_root, "anomaly_detection_temp_ALL_results.json")
    with open(out_all, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nSaved combined results:", out_all)


if __name__ == "__main__":
    main()