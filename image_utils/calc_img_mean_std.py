import cv2
import os
import numpy as np
from tqdm import tqdm
import torch


if __name__ == '__main__':
    data_dir = '../data/Train_Dataset'
    images = os.listdir(data_dir)
    batch_size = len(images)
    h, w = 256, 256
    batch = torch.zeros(batch_size, 3, h, w, dtype=torch.uint8)
    with tqdm(total=batch_size) as pbar:
        for i, img_name in enumerate(images):
            img = cv2.imread(os.path.join(data_dir, img_name))
            img = np.resize(img, (h, w, 3))
            img_t = torch.from_numpy(img)
            img_t = img_t.permute(2, 0, 1)[:3]
            batch[i] = img_t
            pbar.update(1)
    batch = batch.float()
    batch /= 255.
    n_channels = batch.shape[1]
    means, stds = [], []
    for c in range(n_channels):
        mean = torch.mean(batch[:, c])
        means.append(mean)
        std = torch.std(batch[:, c])
        stds.append(std)
    print(means)
    print(stds)

# test dataset
# [tensor(0.0448), tensor(0.0724), tensor(0.1015)]
# [tensor(0.1064), tensor(0.1568), tensor(0.2085)]
#
# train dataset
# [tensor(0.0502), tensor(0.0796), tensor(0.1090)]
# [tensor(0.1184), tensor(0.1700), tensor(0.2183)]
