class Dataset():
    def __init__(self, x, y):
        self.x, self.y = x, y
        assert len(self.x) == len(self.y), "Tensors have different length"

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]
