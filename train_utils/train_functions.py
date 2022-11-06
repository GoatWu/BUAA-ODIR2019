import csv
import sys
from tqdm import tqdm
import torch
from torch.utils import data
from torch.nn import functional as F


def read_csv_data(annotation_file):
    result = {}
    with open(annotation_file) as csvDataFile:
        csv_reader = csv.reader(csvDataFile)
        for row in csv_reader:
            id, N, D, G, C, A, H, M, Oth = row[0:9]
            if id != "id":
                result[id] = [N, D, G, C, A, H, M, Oth]
    return result


def generate_train_valid_set(dataset, split_ratio):
    lenth = dataset.__len__()
    valid_lenth = int(lenth * split_ratio)
    train_lenth = lenth - valid_lenth
    trainset, validset = data.random_split(dataset, [train_lenth, valid_lenth])
    return trainset, validset


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, dt in enumerate(data_loader):
        images, labels = dt
        labels = labels.float().to(device)
        pred = model(images.to(device))
        loss = torch.tensor(0, dtype=torch.float).to(device)
        for i in range(labels.shape[1]):
            loss += F.binary_cross_entropy_with_logits(pred[i, :].squeeze(), labels[:, i])
        loss.backward()
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses
        data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        optimizer.step()
        optimizer.zero_grad()
    return mean_loss.item()


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    total_num = torch.zeros(1).to(device)
    pred_num = torch.zeros(1).to(device)
    TP_num = torch.zeros(1).to(device)
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        total_num += labels.sum()
        pred = model(images.to(device))
        pred = pred.squeeze().permute(1, 0) > 0.5
        pred_num += pred.sum()
        TP_num += torch.logical_and(pred, labels.to(device)).sum()
    precision = TP_num.item() / pred_num.item()
    recall = TP_num.item() / total_num.item()
    return precision, recall
