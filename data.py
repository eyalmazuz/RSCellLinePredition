import torch
from torch.utils.data import Dataset


class IC50Dataset(Dataset):
    def __init__(self, df, labels, categorical_columns, uniques, numerical_columns=None):
        self.df = df.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns

        uniques = [item for sublist in uniques for item in sublist]
        self.num_uniques = len(uniques)
        print(f'{self.num_uniques=}')

        self.cat2id = {cat: i + 1 for i, cat in enumerate(uniques)}
        self.cat2id['<UNK>'] = 0

        self.id2cat = {i: cat for i, cat in enumerate(self.cat2id)}

    def __len__(self,):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        categoricals = row[self.categorical_columns].apply(lambda id: self.cat2id.get(id, 0))

        categoricals = torch.tensor(categoricals.tolist(), dtype=torch.int64)

        if self.numerical_columns:
            numericals = row[self.numerical_columns]
            numericals = torch.tensor(numericals.tolist(), dtype=torch.float32)
            return categoricals, numericals, label

        return categoricals, label
