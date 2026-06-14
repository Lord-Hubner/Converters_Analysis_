class NCTDConverter:
    def __init__(self):
        pass

    def minmax_fit(self, X):
        if isinstance(X, torch.Tensor):
            X = X.numpy()

        self.xmin = X.min(axis=0)
        self.xmax = X.max(axis=0)

    def transform_row(self, x):
        # normalize
        x = (x - self.xmin) / (self.xmax - self.xmin + 1e-8)

        n = len(x)
        N = int(np.ceil(np.sqrt(n)))

        padded = np.zeros(N*N)

       # for i in range(N):
        #    padded[i] = np.roll(base) 

        # pad to square
        padded[:n] = x

        base = padded.reshape(N, N)

        # geometric transforms
        hflip = np.fliplr(base)
        vflip = np.flipud(base)
        rot = np.rot90(base)

        # build 2N x 2N image
        top = np.concatenate([base, hflip], axis=1)
        bottom = np.concatenate([vflip, rot], axis=1)

        img = np.concatenate([top, bottom], axis=0)

        return img.astype(np.float32)

    def transform(self, X):
        return np.stack([self.transform_row(x) for x in X])