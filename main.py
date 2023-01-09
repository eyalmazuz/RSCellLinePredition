import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import DataLoader

import wandb

from data import IC50Dataset
from model import FM, DeepFM
from preprocess import extract_molecule_features
from train import train, validation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def fit_model(df, cat_columns, num_columns, k, epochs, batch_size):
    labels = df['LN_IC50']
    df = df.drop(columns=['LN_IC50'])

    X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.2, shuffle=True)

    run = wandb.init(project='IC50', job_type='Experiments',
                notes='Testing whether DFM can predict IC50 from cancer data',
                tags=['FM', 'Research', 'GroupCV'],  # + [' & '.join(X_test['TCGA_DESC'].astype(str).unique())],
                reinit=True)

    uniques = []
    for col in cat_columns:
        uniques.append(X_train[col].unique().tolist())

    cat_dims = list(map(len, uniques))
    train_dataset = IC50Dataset(X_train, y_train, cat_columns, uniques, num_columns)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = IC50Dataset(X_test, y_test, cat_columns, uniques, num_columns)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f'{len(train_dataset)=}, {len(val_dataset)=}')

    model = FM(cat_dims, len(num_columns), k=k).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        train(model, train_loader, optimizer, criterion, epoch, run, device)
        validation(model, val_loader, epoch, run, device)

    run.finish()


def main():
    df = pd.read_csv('../../data/GDSC2_fitted_dose_response_24Jul22.csv', usecols=['DRUG_NAME', 'PATHWAY_NAME',
                                                                                   'CELL_LINE_NAME', 'PUTATIVE_TARGET',
                                                                                   'MIN_CONC', 'MAX_CONC',
                                                                                   'LN_IC50'])
    smiles = pd.read_csv('../../data/drugs.csv')

    df = extract_molecule_features(df, smiles)

    df.to_csv('./features_ic50.csv', index=False)

    fit_model(df, ['DRUG_NAME', 'PATHWAY_NAME', 'CELL_LINE_NAME', 'PUTATIVE_TARGET'],
              ['MIN_CONC', 'MAX_CONC',
               'MolWt', 'TPSA', 'LogP', 'HAcceptors',
               'HDonors', 'RotatableBonds', 'RingCount', 'AromaticRings',
               'RingCount', 'ALERTS'],
              k=64, epochs=1000, batch_size=256)


if __name__ == "__main__":
    # set_seed(42)
    main()
