import os
import numpy as np
import pandas as pd

from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold

import pyDeepInsight


class NCTDConverter:
    """
    Novel Algorithm for Convolving Tabular Data (NCTD)

    Transforming tabular data into images via enhanced
    spatial relationships for CNN processing.
    """

    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit(self, X):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        X = self.scaler.transform(X)
        images = []

        for row in X:
            row = np.round(row * 255).astype(np.uint8)
            n = len(row)

            # N x N rotated matrix
            A = np.vstack([
                np.roll(row, shift)
                for shift in range(n)
            ])

            # 2N x 2N expansion
            img = np.block([
                [A, A],
                [A, A]
            ])

            images.append(img)

        return np.asarray(images)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def build_converter(converter_name):
    if converter_name == "NCTD":
        return NCTDConverter()

    if converter_name == "DeepInsight":
        tsne = TSNE(n_components=2, perplexity=3, random_state=42)
        return pyDeepInsight.image_transformer.ImageTransformer(
            tsne,
            "bin",
            (32, 32)
        )

    raise ValueError(f"Unknown converter: {converter_name}")


def to_uint8_image(img):
    img = np.asarray(img)

    if img.dtype == np.uint8:
        return img

    img = np.nan_to_num(img)
    img_min = img.min()
    img_max = img.max()

    if img_min < 0 or img_max > 255:
        img = (img - img_min) / (img_max - img_min + 1e-8)

    if img.max() <= 1:
        img = img * 255

    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    INPUT_FILE = "KIRC_features_nogene.csv"
    OUTPUT_DIR = "NCTD_KIRC_IMGS"
    CONVERTER = "NCTD"
    MAKE_FOLDS = False
    N_SPLITS = 5

    print("Loading dataset...")

    df = pd.read_csv(INPUT_FILE, index_col=0)
    arenull = df.isnull().sum()
    print(arenull)

    y = df["label"].values
    X = df.drop(columns=["label"]).values

    print(f"Samples: {len(X)}")
    print(f"Features: {X.shape[1]}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    assignment_path = os.path.join(OUTPUT_DIR, "fold_assignments.csv")
    assignments = None

    if MAKE_FOLDS:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        fold_assignments = []

        for fold, (train_idx, _) in enumerate(skf.split(X, y), start=1):
            for idx in train_idx:
                fold_assignments.append({
                    "sample_idx": int(idx),
                    "fold": fold
                })

        assignments = pd.DataFrame(fold_assignments)
        assignments.to_csv(assignment_path, index=False)
    else:
        if not os.path.exists(assignment_path):
            raise FileNotFoundError(
                f"MAKE_FOLDS is False, but no assignments file was found at {assignment_path}"
            )
        assignments = pd.read_csv(assignment_path)

    required_columns = {"sample_idx", "fold"}
    if not required_columns.issubset(assignments.columns):
        raise ValueError(
            f"{assignment_path} must contain columns: {sorted(required_columns)}"
        )

    assignments["sample_idx"] = assignments["sample_idx"].astype(int)
    assignments["fold"] = assignments["fold"].astype(int)

    folds_idxs = {
        fold: assignments.loc[assignments["fold"] == fold, "sample_idx"].to_numpy()
        for fold in sorted(assignments["fold"].unique())
    }

    print(f"Generating {CONVERTER} images...")

    class_map = {
        0: "Upregulated",
        1: "Downregulated",
        2: "NoDifference"
    }

    all_metadata = []

    for fold_number, fold_indices in folds_idxs.items():
        fold_output = os.path.join(OUTPUT_DIR, "fold_" + str(fold_number))
        os.makedirs(fold_output, exist_ok=True)

        current_X = X[fold_indices]
        current_y = y[fold_indices]

        converter = build_converter(CONVERTER)
        images = converter.fit_transform(current_X)

        print(f"Fold {fold_number} generated shape: {images.shape}")

        metadata = []

        for idx, (sample_idx, img, label) in enumerate(zip(fold_indices, images, current_y)):
            label_value = int(label)
            class_name = class_map[label_value]

            class_dir = os.path.join(fold_output, class_name)
            os.makedirs(class_dir, exist_ok=True)

            filename = f"sample_{sample_idx:06d}.png"
            filepath = os.path.join(class_dir, filename)

            Image.fromarray(to_uint8_image(img)).save(filepath)

            row_metadata = {
                "fold": fold_number,
                "sample_idx": int(sample_idx),
                "filename": filename,
                "label": label_value,
                "class": class_name,
                "path": filepath
            }

            metadata.append(row_metadata)
            all_metadata.append(row_metadata)

            if idx % 100 == 0:
                print(f"Saved {idx}/{len(images)} images")

        pd.DataFrame(metadata).to_csv(
            os.path.join(fold_output, "labels.csv"),
            index=False
        )

    pd.DataFrame(all_metadata).to_csv(
        os.path.join(OUTPUT_DIR, "labels.csv"),
        index=False
    )

    print("\nFinished!")
    print(f"Images saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
