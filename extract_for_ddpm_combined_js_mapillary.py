import os
import json
import shutil
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


ADV_ATTACKS = [
    "fgsm",
    "rfgsm",
    "pgd",
    "random_patch",
]

ENV_ATTACKS = [
    "gaussian",
    "salt_pepper",
    "light",
    "fog",
    "motion_blur",
]

DEFAULT_ATTACKS = ADV_ATTACKS + ENV_ATTACKS

COMBINED_ROOT = "./combined_detector_js_calibrated_mapillary"

MODELS = {
    "mobilenet": {
        "adv_root": "./attacks_mobilenet_eps",
        "env_root": "./env_mobilenet",
        "eval_root": "./eval_mobilenet",
        "out_root": "./ddpm_input_combined_js_mobilenet",
    },
    "convnext": {
        "adv_root": "./attacks_convnext_eps",
        "env_root": "./env_convnext",
        "eval_root": "./eval_convnext",
        "out_root": "./ddpm_input_combined_js_convnext",
    },
    "efficientnet": {
        "adv_root": "./attacks_efficientnet_eps",
        "env_root": "./env_efficientnet",
        "eval_root": "./eval_efficientnet",
        "out_root": "./ddpm_input_combined_js_efficientnet",
    },
}


def png_dir_for(cfg, attack):
    if attack in ADV_ATTACKS:
        return os.path.join(cfg["adv_root"], f"{attack}_png")
    if attack in ENV_ATTACKS:
        return os.path.join(cfg["env_root"], f"{attack}_png")
    raise ValueError(f"Unknown condition: {attack}")


def clean_attack_output(out_dir):
    """Rebuild ONLY the new combined-JS output directory for this condition."""
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)


def ensure_under_root(path, root):
    abs_path = os.path.abspath(os.path.realpath(path))
    abs_root = os.path.abspath(os.path.realpath(root))

    try:
        common = os.path.commonpath([abs_path, abs_root])
    except ValueError as exc:
        raise RuntimeError(
            f"Cannot compare paths:\n  path={abs_path}\n  root={abs_root}"
        ) from exc

    if common != abs_root:
        raise RuntimeError(
            "Evaluation filename does not belong to the expected PNG directory.\n"
            f"  file: {abs_path}\n"
            f"  expected root: {abs_root}\n"
            "This may indicate stale or mismatched evaluation outputs."
        )

    rel_path = os.path.relpath(abs_path, abs_root)
    if rel_path.startswith(".."):
        raise RuntimeError(f"Unsafe relative path produced: {rel_path}")

    return abs_path, rel_path


def load_final_flagged_pairs(model_key, cfg, attack, png_dir):
    """Load final OLD OR calibrated-JS mask and map mask positions to PNG files."""

    combined_dir = os.path.join(
        COMBINED_ROOT,
        model_key,
        attack,
    )

    mask_path = os.path.join(
        combined_dir,
        "calibrated_combined_or_mask.npy",
    )

    eval_dir = os.path.join(
        cfg["eval_root"],
        attack,
    )

    filenames_path = os.path.join(
        eval_dir,
        "filenames.npy",
    )

    if not os.path.exists(mask_path):
        raise FileNotFoundError(
            f"Missing final combined mask: {mask_path}\n"
            "Run the joint OLD+JS calibration script first."
        )

    if not os.path.exists(filenames_path):
        raise FileNotFoundError(
            f"Missing evaluation filenames: {filenames_path}"
        )

    mask = np.load(mask_path, allow_pickle=False)
    filenames = np.load(filenames_path, allow_pickle=True)

    mask = np.asarray(mask, dtype=bool).reshape(-1)
    filenames = np.asarray(filenames, dtype=object).reshape(-1)

    if len(mask) != len(filenames):
        raise RuntimeError(
            f"[{model_key}/{attack}] mask/filename length mismatch:\n"
            f"  mask={len(mask)}\n"
            f"  filenames={len(filenames)}"
        )

    flagged_indices = np.flatnonzero(mask).astype(np.int64)

    pairs = []
    for idx in flagged_indices:
        source_from_eval = str(filenames[int(idx)])
        abs_src, rel_path = ensure_under_root(source_from_eval, png_dir)
        pairs.append((abs_src, rel_path))

    return pairs, len(filenames), len(flagged_indices)


def copy_pairs(pairs, out_dir, copy_images, model_key, attack):
    copied = 0
    missing = 0

    files_txt = os.path.join(out_dir, "files.txt")

    with open(files_txt, "w") as f:
        for src, rel_path in tqdm(
            pairs,
            desc=f"  {model_key}/{attack}",
            leave=False,
        ):
            f.write(rel_path + "\n")

            if not copy_images:
                continue

            if not os.path.exists(src):
                missing += 1
                continue

            dst = os.path.join(out_dir, rel_path)
            Path(os.path.dirname(dst)).mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1

    return copied, missing


def extract_attack(model_key, cfg, attack, copy_images):
    png_dir = png_dir_for(cfg, attack)

    if not os.path.isdir(png_dir):
        print(f"  [{attack}] SKIP - missing PNG directory: {png_dir}")
        return None

    out_dir = os.path.join(cfg["out_root"], attack)

    # IMPORTANT: this deletes/rebuilds ONLY the NEW combined-JS extraction
    # directory. Old ddpm_input_<model>/ folders are untouched.
    clean_attack_output(out_dir)

    pairs, n_total, n_selected = load_final_flagged_pairs(
        model_key=model_key,
        cfg=cfg,
        attack=attack,
        png_dir=png_dir,
    )

    copied, missing = copy_pairs(
        pairs=pairs,
        out_dir=out_dir,
        copy_images=copy_images,
        model_key=model_key,
        attack=attack,
    )

    if missing > 0:
        raise RuntimeError(
            f"{model_key}/{attack}: {missing} selected source files were missing."
        )

    stats = {
        "model": model_key,
        "attack": attack,
        "mode": "final_calibrated_old_or_js",
        "n_total_evaluated": int(n_total),
        "n_selected": int(n_selected),
        "selected_fraction_percent": float(
            100.0 * n_selected / n_total if n_total else 0.0
        ),
        "copied": int(copied if copy_images else 0),
        "missing": int(missing),
        "copy_images": bool(copy_images),
        "source_png_dir": png_dir,
        "combined_mask": os.path.join(
            COMBINED_ROOT,
            model_key,
            attack,
            "calibrated_combined_or_mask.npy",
        ),
        "out_dir": out_dir,
        "files_list": os.path.join(out_dir, "files.txt"),
    }

    with open(
        os.path.join(out_dir, "extraction_stats.json"),
        "w",
    ) as f:
        json.dump(stats, f, indent=2)

    print(
        f"  [{attack:13s}] "
        f"selected={n_selected:6d}/{n_total:6d}  "
        f"copied={copied:6d}  missing={missing}"
    )

    return stats


def extract_model(model_key, attacks, copy_images):
    cfg = MODELS[model_key]
    Path(cfg["out_root"]).mkdir(parents=True, exist_ok=True)

    print(
        f"\n[{model_key}] FINAL CALIBRATED OLD+JS FLAGGED ONLY "
        f"-> {cfg['out_root']}"
    )

    all_stats = []

    for attack in attacks:
        stats = extract_attack(
            model_key=model_key,
            cfg=cfg,
            attack=attack,
            copy_images=copy_images,
        )
        if stats is not None:
            all_stats.append(stats)

    summary = {
        "model": model_key,
        "mode": "final_calibrated_old_or_js",
        "copy_images": bool(copy_images),
        "attacks_requested": attacks,
        "by_attack": all_stats,
    }

    with open(
        os.path.join(cfg["out_root"], "extraction_summary.json"),
        "w",
    ) as f:
        json.dump(summary, f, indent=2)

    return all_stats


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()) + ["all"],
        required=True,
    )

    parser.add_argument(
        "--attacks",
        nargs="+",
        choices=DEFAULT_ATTACKS,
        default=DEFAULT_ATTACKS,
        help="Conditions to extract. Clean is intentionally excluded.",
    )

    parser.add_argument(
        "--no_copy",
        action="store_true",
        help="Write file lists/statistics only; do not copy PNGs.",
    )

    args = parser.parse_args()
    copy_images = not args.no_copy

    model_keys = (
        list(MODELS.keys())
        if args.model == "all"
        else [args.model]
    )

    print("=" * 96)
    print("EXTRACT FINAL CALIBRATED OLD+JS FLAGGED IMAGES FOR DDPM")
    print("=" * 96)
    print(f"Models             : {model_keys}")
    print(f"Conditions         : {args.attacks}")
    print(f"Combined mask root : {COMBINED_ROOT}")
    print("Old DDPM inputs    : PRESERVED / NOT MODIFIED")
    print("New output roots   : ddpm_input_combined_js_<model>/")
    print(f"Copy files         : {copy_images}")

    grand = {}

    for model_key in model_keys:
        grand[model_key] = extract_model(
            model_key=model_key,
            attacks=args.attacks,
            copy_images=copy_images,
        )

    print("\n" + "=" * 96)
    print("SUMMARY: FINAL COMBINED-DETECTOR FLAGGED COUNTS")
    print("=" * 96)

    header = "model        " + "  ".join(
        f"{attack[:10]:>10s}" for attack in args.attacks
    )
    print(header)

    for model_key in model_keys:
        by_attack = {
            stat["attack"]: stat
            for stat in grand[model_key]
        }

        row = f"{model_key:13s}"

        for attack in args.attacks:
            stat = by_attack.get(attack)
            if stat is None:
                row += f"  {'--':>10s}"
            else:
                row += f"  {stat['n_selected']:10d}"

        print(row)

    print("\nDone.")
    print("Old ddpm_input_<model>/ folders were not touched.")
    print("Use ddpm_input_combined_js_<model>/ for the new reconstruction run.")


if __name__ == "__main__":
    main()