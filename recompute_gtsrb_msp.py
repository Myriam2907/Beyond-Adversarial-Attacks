

import os
import numpy as np

ROOT = "./gtsrb_repeat/signals"

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
    "patch",
    "random_patch",
    "gaussian_noise",
    "salt_pepper",
    "light",
    "fog",
    "motion_blur",
]


def load_array(model, condition, name):
    path = os.path.join(ROOT, model, condition, f"{name}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path)


for model in MODELS:

    print("\n" + "=" * 100)
    print(f"MODEL: {model}")
    print("=" * 100)

   
    clean_conf = load_array(model, "clean", "confidence")

    msp_threshold = np.percentile(clean_conf, 5)

    print(f"MSP threshold (clean 5th percentile): {msp_threshold:.8f}")
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

        conf = load_array(model, condition, "confidence")
        pred = load_array(model, condition, "pred")
        label = load_array(model, condition, "label")

        assert len(conf) == len(pred) == len(label)

        
        correct = pred == label
        wrong = ~correct

        accuracy = 100.0 * correct.mean()

        
        detected = conf < msp_threshold

        n_detected = int(detected.sum())
        n_wrong = int(wrong.sum())

        if condition == "clean":

            fpr = 100.0 * detected.mean()

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

            
            tpr = 100.0 * detected.mean()

            
            if n_wrong > 0:
                ecdr = 100.0 * (detected & wrong).sum() / n_wrong
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