import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
from pyDeepInsight import image_transformer
from sklearn.manifold import TSNE

class WDBCDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)

        self.features = torch.tensor(self.data.iloc[:, 2:].values, dtype=torch.float32)

        encoder = LabelEncoder()
        targets = encoder.fit_transform(self.data.iloc[:, 1].values)

        self.targets = torch.tensor(targets, dtype=torch.long)
        self.transform= transform

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        feature = self.features[idx]
        target = self.targets[idx]

        if self.transform:
            feature = self.transform(feature)
            feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)
        return feature, target

class NCTDConverter:
    """
    Novel Algorithm for Convolving Tabular Data (NCTD)

    Paper:
    'Transforming tabular data into images via enhanced spatial
    relationships for CNN processing'
    """

    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit(self, X: np.ndarray):
        self.scaler.fit(X)
        return self

    def transform(self, X: np.ndarray):
        X = self.scaler.transform(X)

        images = []

        for row in X:
            row = np.round(row * 255).astype(np.uint8)
            n = len(row)

            # N x N rotated matrix
            A = np.vstack([
                np.roll(row, +shift)
                for shift in range(n)
            ])

            # 2N x 2N expansion
            img = np.block([
                [A, A],
                [A, A]
            ])

            images.append(img.astype(np.float32))

        return np.asarray(images)

    def fit_transform(self, X: np.ndarray):
        self.fit(X)
        return self.transform(X)

def main():
    converter = NCTDConverter()
    dataset = WDBCDataset("wdbc.data")

    #fe = TSNE(perplexity=10)
    #it = image_transformer.ImageTransformer(feature_extractor=fe) 

    #print(dataset.features.shape)

    #x_img = it.fit_transform(dataset.features)


    x_imgs = converter.fit_transform(dataset.features)

    counter = 0
    for img in x_imgs:
        print(f"Current label is {dataset.targets[counter]}")
        plt.imshow(img, cmap="viridis")
        plt.colorbar()
        plt.show()
        counter+=1

  #  for x, y in x_img:
  #      img = x[0][0]
  #      print(img)
  #      print(img.shape)
  #      plt.imshow(img, cmap="gray", interpolation="nearest")
  #      plt.grid(True)
  #      plt.show()

    
if __name__ == "__main__":
    main()
