

import os
import numpy as np

ROOT = "./js_consistency_signal_mapillary"

MODELS = [
    "mobilenet",
    "convnext",
    "efficientnet",
]

CONDITIONS = [
    "clean",
    "fgsm",
    "rfgsm",
    "pgd",
    "random_patch",
    "gaussian",
    "salt_pepper",
    "light",
    "fog",
    "motion_blur",
]


def load_array(model, condition, name):
    path = os.path.join(ROOT, model, condition, f"{name}.npy")

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing: {path}")

    return np.load(path)


print("=" * 110)
print("FRESH MTSD / MAPILLARY MSP EVALUATION")
print("=" * 110)

for model in MODELS:

    print("\n" + "=" * 110)
    print(f"MODEL: {model}")
    print("=" * 110)

   

    clean_conf = load_array(
        model,
        "clean",
        "confidence_base",
    )

    msp_threshold = np.percentile(clean_conf, 5)

    print(
        f"MSP threshold "
        f"(clean 5th percentile): "
        f"{msp_threshold:.8f}"
    )

    print()

    print(
        f"{'Condition':<18}"
        f"{'Acc':>10}"
        f"{'FPR':>10}"
        f"{'TPR':>10}"
        f"{'ECDR':>12}"
        f"{'Detected':>12}"
        f"{'Wrong':>12}"
    )

    print("-" * 84)

    for condition in CONDITIONS:

        conf = load_array(
            model,
            condition,
            "confidence_base",
        )

        pred = load_array(
            model,
            condition,
            "pred_base",
        )

        label = load_array(
            model,
            condition,
            "labels",
        )

        if not (
            len(conf)
            == len(pred)
            == len(label)
        ):
            raise RuntimeError(
                f"Array length mismatch: "
                f"{model}/{condition}"
            )

        
        correct = pred == label
        wrong = ~correct

        accuracy = 100.0 * np.mean(correct)

      

        detected = conf < msp_threshold

        n_detected = int(np.sum(detected))
        n_wrong = int(np.sum(wrong))

        if condition == "clean":

            fpr = 100.0 * np.mean(detected)

            print(
                f"{condition:<18}"
                f"{accuracy:>9.2f}%"
                f"{fpr:>9.2f}%"
                f"{'N/A':>10}"
                f"{'N/A':>12}"
                f"{n_detected:>12}"
                f"{n_wrong:>12}"
            )

        else:

            tpr = 100.0 * np.mean(detected)

            
            if n_wrong > 0:
                ecdr = (
                    100.0
                    * np.sum(detected & wrong)
                    / n_wrong
                )
            else:
                ecdr = 0.0

            print(
                f"{condition:<18}"
                f"{accuracy:>9.2f}%"
                f"{'N/A':>10}"
                f"{tpr:>9.2f}%"
                f"{ecdr:>11.2f}%"
                f"{n_detected:>12}"
                f"{n_wrong:>12}"
            )

print("\n" + "=" * 110)
print("DONE")
print("=" * 110)