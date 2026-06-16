"""
Script para calcular a média das métricas de teste por {conversor}_{dataset}
a partir dos arquivos metrics.json de cada fold.
"""

import json
import os
import glob
from collections import defaultdict

RESULTS_DIR = "cnn_results"

def main():
    groups = defaultdict(list)

    for fold_dir in sorted(glob.glob(os.path.join(RESULTS_DIR, "*_fold_*"))):
        metrics_file = os.path.join(fold_dir, "metrics.json")
        if not os.path.isfile(metrics_file):
            continue

        with open(metrics_file, "r") as f:
            data = json.load(f)

        converter = data["converter"]
        dataset = data["dataset"]
        key = f"{converter}_{dataset}"
        groups[key].append(data["test"])

    print(f"{'Conversor_Dataset':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'ROC-AUC':>10} {'Loss':>10} {'Folds':>6}")
    print("-" * 95)

    results = {}
    for key in sorted(groups.keys()):
        folds = groups[key]
        n = len(folds)
        metrics_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "loss"]
        avg = {m: sum(f[m] for f in folds) / n for m in metrics_keys}
        results[key] = {"mean": avg, "n_folds": n}

        print(f"{key:<25} {avg['accuracy']:>10.4f} {avg['precision']:>10.4f} {avg['recall']:>10.4f} "
              f"{avg['f1_score']:>10.4f} {avg['roc_auc']:>10.4f} {avg['loss']:>10.4f} {n:>6}")

    output_file = os.path.join(RESULTS_DIR, "average_metrics.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResultados salvos em: {output_file}")


if __name__ == "__main__":
    main()
