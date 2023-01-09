from sklearn.metrics import mean_absolute_error, r2_score
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F

import wandb


def train(model, dataloader, optimizer, criterion, epoch, run, device):
    model.train()
    for step, (*batch, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch = [data.to(device) for data in batch]
        labels = labels.to(device)

        preds = model(*batch)

        loss = criterion(preds.squeeze(), labels)

        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        if step % 100 == 0:
            r2 = r2_score(labels.cpu().numpy(), preds.squeeze().detach().cpu().numpy())
            mae = mean_absolute_error(labels.cpu().numpy(), preds.squeeze().detach().cpu().numpy())

            run.log({'train/loss': loss.item(),
                     'train/mae': mae,
                     'train/r2': r2,
                     'epoch': epoch,
                     'step': epoch * len(dataloader) + step})


@torch.no_grad()
def validation(model, dataloader, epoch, run, device):
    model.eval()
    preds = []
    labels = []
    for step, (*batch, label) in tqdm(enumerate(dataloader), total=len(dataloader), leave=False):
        batch = [data.to(device) for data in batch]
        labels += label.numpy().tolist()

        pred = model(*batch)

        preds += pred.cpu().numpy().tolist()

    loss = F.mse_loss(torch.tensor(preds).squeeze(), torch.tensor(labels))
    r2 = r2_score(labels, preds)
    mae = mean_absolute_error(labels, preds)

    run.log({'eval/loss': loss.item(),
               'eval/mae': mae,
               'eval/r2': r2,
               'epoch': epoch, })

