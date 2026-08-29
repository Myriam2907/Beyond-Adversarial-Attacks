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

DEFAULT_ATTACKS = (
    ADV_ATTACKS +
    ENV_ATTACKS
)

MODELS = {
    "mobilenet": {
        "adv_root": "./attacks_mobilenet_eps",
        "env_root": "./env_mobilenet",
        "eval_root": "./eval_mobilenet",
        "out_root": "./ddpm_input_mobilenet",
    },
    "convnext": {
        "adv_root": "./attacks_convnext_eps",
        "env_root": "./env_convnext",
        "eval_root": "./eval_convnext",
        "out_root": "./ddpm_input_convnext",
    },
    "efficientnet": {
        "adv_root": "./attacks_efficientnet_eps",
        "env_root": "./env_efficientnet",
        "eval_root": "./eval_efficientnet",
        "out_root": "./ddpm_input_efficientnet",
    },
}




def png_dir_for(cfg, attack):
    if attack in ADV_ATTACKS:
        return os.path.join(
            cfg["adv_root"],
            f"{attack}_png"
        )

    if attack in ENV_ATTACKS:
        return os.path.join(
            cfg["env_root"],
            f"{attack}_png"
        )

    raise ValueError(
        f"Unknown attack/corruption: {attack}"
    )


def list_all_pngs(png_dir):
   
    rels = []

    for root, dirs, files in os.walk(png_dir):
        dirs.sort()
        files.sort()

        for filename in files:
            if not filename.lower().endswith(".png"):
                continue

            full_path = os.path.join(
                root,
                filename
            )

            rel_path = os.path.relpath(
                full_path,
                png_dir
            )

            rels.append(
                rel_path
            )

    rels.sort()

    return rels


def clean_attack_output(out_dir):
    
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    Path(out_dir).mkdir(
        parents=True,
        exist_ok=True
    )


def ensure_under_root(path, root):
   
    abs_path = os.path.abspath(
        os.path.realpath(path)
    )

    abs_root = os.path.abspath(
        os.path.realpath(root)
    )

    try:
        common = os.path.commonpath([
            abs_path,
            abs_root
        ])
    except ValueError as exc:
        raise RuntimeError(
            f"Cannot compare paths:\n"
            f"  path={abs_path}\n"
            f"  root={abs_root}"
        ) from exc

    if common != abs_root:
        raise RuntimeError(
            "Detector filename does not belong to the expected "
            "attack/corruption directory.\n"
            f"  file: {abs_path}\n"
            f"  expected root: {abs_root}\n"
            "This may indicate stale or mismatched eval outputs."
        )

    rel_path = os.path.relpath(
        abs_path,
        abs_root
    )

    if rel_path.startswith(".."):
        raise RuntimeError(
            f"Unsafe relative path produced: {rel_path}"
        )

    return abs_path, rel_path


def load_flagged_pairs(
    cfg,
    attack,
    png_dir
):
    
    eval_dir = os.path.join(
        cfg["eval_root"],
        attack
    )

    idx_path = os.path.join(
        eval_dir,
        "suspicious_indices.npy"
    )

    filenames_path = os.path.join(
        eval_dir,
        "filenames.npy"
    )

    if not os.path.isdir(eval_dir):
        raise FileNotFoundError(
            f"Missing evaluation directory: {eval_dir}\n"
            "Run detector evaluation first."
        )

    if not os.path.exists(idx_path):
        raise FileNotFoundError(
            f"Missing detector output: {idx_path}\n"
            "Run:\n"
            "  python detect_unified_v2.py --model <model>"
        )

    if not os.path.exists(filenames_path):
        raise FileNotFoundError(
            f"Missing filenames array: {filenames_path}\n"
            "Regenerate evaluation features with:\n"
            "  python eval_attacked_unified_v2.py --model <model>"
        )

    suspicious_indices = np.load(
        idx_path,
        allow_pickle=False
    )

    filenames = np.load(
        filenames_path,
        allow_pickle=True
    )

    suspicious_indices = np.asarray(
        suspicious_indices,
        dtype=np.int64
    ).reshape(-1)

    filenames = np.asarray(
        filenames,
        dtype=object
    ).reshape(-1)

    n_total = len(
        filenames
    )

    if n_total == 0:
        raise RuntimeError(
            f"No filenames found in {filenames_path}"
        )

    if suspicious_indices.size:
        if suspicious_indices.min() < 0:
            raise RuntimeError(
                f"Negative suspicious index found in {idx_path}"
            )

        if suspicious_indices.max() >= n_total:
            raise RuntimeError(
                "Suspicious-index mismatch:\n"
                f"  max index = {suspicious_indices.max()}\n"
                f"  n filenames = {n_total}\n"
                f"  file = {idx_path}"
            )

    
    suspicious_indices = np.unique(
        suspicious_indices
    )

    pairs = []

    for idx in suspicious_indices:
        source_from_eval = str(
            filenames[
                int(idx)
            ]
        )

        abs_src, rel_path = ensure_under_root(
            source_from_eval,
            png_dir
        )

        pairs.append(
            (
                abs_src,
                rel_path
            )
        )

    return (
        pairs,
        n_total,
        len(pairs)
    )


def load_all_pairs(
    png_dir
):
    rels = list_all_pngs(
        png_dir
    )

    pairs = [
        (
            os.path.abspath(
                os.path.join(
                    png_dir,
                    rel_path
                )
            ),
            rel_path
        )
        for rel_path in rels
    ]

    return pairs


def copy_pairs(
    pairs,
    out_dir,
    copy_images,
    model_key,
    attack
):
    
    copied = 0
    missing = 0

    list_path = os.path.join(
        out_dir,
        "files.txt"
    )

    with open(
        list_path,
        "w"
    ) as file_list:

        for src, rel_path in tqdm(
            pairs,
            desc=f"  {model_key}/{attack}",
            leave=False
        ):
            file_list.write(
                rel_path +
                "\n"
            )

            if not copy_images:
                continue

            if not os.path.exists(src):
                missing += 1
                continue

            dst = os.path.join(
                out_dir,
                rel_path
            )

            Path(
                os.path.dirname(dst)
            ).mkdir(
                parents=True,
                exist_ok=True
            )

            shutil.copy2(
                src,
                dst
            )

            copied += 1

    return copied, missing




def extract_attack(
    model_key,
    cfg,
    attack,
    only_flagged,
    copy_images
):
    png_dir = png_dir_for(
        cfg,
        attack
    )

    out_dir = os.path.join(
        cfg["out_root"],
        attack
    )

    if not os.path.isdir(
        png_dir
    ):
        print(
            f"  [{attack}] SKIP - missing PNG directory: "
            f"{png_dir}"
        )

        return None


    clean_attack_output(
        out_dir
    )

    if only_flagged:
        pairs, n_total, n_selected = load_flagged_pairs(
            cfg=cfg,
            attack=attack,
            png_dir=png_dir
        )

        source_desc = (
            f"{n_selected}/{n_total} flagged"
        )

    else:
        pairs = load_all_pairs(
            png_dir
        )

        n_total = len(
            pairs
        )

        n_selected = n_total

        source_desc = (
            f"{n_selected} all"
        )

    copied, missing = copy_pairs(
        pairs=pairs,
        out_dir=out_dir,
        copy_images=copy_images,
        model_key=model_key,
        attack=attack
    )

    if missing > 0:
        raise RuntimeError(
            f"{model_key}/{attack}: "
            f"{missing} selected source files were missing."
        )

    stats = {
        "model": model_key,
        "attack": attack,
        "mode": (
            "only_flagged"
            if only_flagged
            else "all"
        ),
        "n_total_evaluated_or_available": int(
            n_total
        ),
        "n_selected": int(
            n_selected
        ),
        "selected_fraction_percent": float(
            (
                100.0 *
                n_selected /
                n_total
            )
            if n_total > 0
            else 0.0
        ),
        "copied": int(
            copied
            if copy_images
            else 0
        ),
        "missing": int(
            missing
        ),
        "copy_images": bool(
            copy_images
        ),
        "png_dir": png_dir,
        "out_dir": out_dir,
        "files_list": os.path.join(
            out_dir,
            "files.txt"
        ),
    }

    stats_path = os.path.join(
        out_dir,
        "extraction_stats.json"
    )

    with open(
        stats_path,
        "w"
    ) as f:
        json.dump(
            stats,
            f,
            indent=2
        )

    print(
        f"  [{attack:13s}] "
        f"{source_desc:>20s}  "
        f"copied={copied:6d}  "
        f"missing={missing}"
    )

    return stats



def extract_model(
    model_key,
    attacks,
    only_flagged,
    copy_images
):
    cfg = MODELS[
        model_key
    ]

    Path(
        cfg["out_root"]
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"\n[{model_key}] "
        f"mode={'FLAGGED ONLY' if only_flagged else 'ALL ATTACKED'} "
        f"out={cfg['out_root']}"
    )

    all_stats = []

    for attack in attacks:
        stats = extract_attack(
            model_key=model_key,
            cfg=cfg,
            attack=attack,
            only_flagged=only_flagged,
            copy_images=copy_images
        )

        if stats is not None:
            all_stats.append(
                stats
            )

    summary = {
        "model": model_key,
        "mode": (
            "only_flagged"
            if only_flagged
            else "all"
        ),
        "copy_images": bool(
            copy_images
        ),
        "attacks_requested": attacks,
        "by_attack": all_stats,
    }

    summary_path = os.path.join(
        cfg["out_root"],
        "extraction_summary.json"
    )

    with open(
        summary_path,
        "w"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2
        )

    return all_stats




def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=list(
            MODELS.keys()
        ) + ["all"],
        required=True
    )

    parser.add_argument(
        "--attacks",
        nargs="+",
        choices=DEFAULT_ATTACKS,
        default=DEFAULT_ATTACKS,
        help=(
            "Conditions to extract. "
            "Clean is intentionally excluded."
        )
    )

    parser.add_argument(
        "--only_flagged",
        action="store_true",
        help=(
            "Extract only images flagged by detect_unified_v2.py. "
            "Default: extract all attacked/corrupted images."
        )
    )

    parser.add_argument(
        "--no_copy",
        action="store_true",
        help=(
            "Write file lists/statistics only; "
            "do not physically copy PNGs."
        )
    )

    args = parser.parse_args()

    copy_images = (
        not args.no_copy
    )

    model_keys = (
        list(
            MODELS.keys()
        )
        if args.model == "all"
        else [
            args.model
        ]
    )

    print(
        "=" * 88
    )

    print(
        "EXTRACT FOR DDPM V2"
    )

    print(
        "=" * 88
    )

    print(
        f"Models     : {model_keys}"
    )

    print(
        f"Attacks    : {args.attacks}"
    )

    print(
        f"Mode       : "
        f"{'FLAGGED ONLY' if args.only_flagged else 'ALL ATTACKED'}"
    )

    print(
        f"Copy files : {copy_images}"
    )

    grand = {}

    for model_key in model_keys:
        grand[
            model_key
        ] = extract_model(
            model_key=model_key,
            attacks=args.attacks,
            only_flagged=args.only_flagged,
            copy_images=copy_images
        )

    print(
        "\n"
        + "=" * 88
    )

    print(
        "SUMMARY (selected/copied per attack)"
    )

    print(
        "=" * 88
    )

    header = (
        "model        "
        + "  ".join(
            f"{attack[:10]:>10s}"
            for attack in args.attacks
        )
    )

    print(
        header
    )

    for model_key in model_keys:
        by_attack = {
            stat["attack"]: stat
            for stat in grand[
                model_key
            ]
        }

        row = f"{model_key:13s}"

        for attack in args.attacks:
            stat = by_attack.get(
                attack
            )

            if stat is None:
                row += (
                    f"  {'--':>10s}"
                )
            else:
                selected = stat[
                    "n_selected"
                ]

                row += (
                    f"  {selected:10d}"
                )

        print(
            row
        )

    print(
        "\nNote: in --only_flagged mode, selected counts should match "
        "the detector's suspicious counts for each condition."
    )


if __name__ == "__main__":
    main()
