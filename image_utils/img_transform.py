import os
import cv2
import numpy as np
from tqdm import tqdm


def deal_a_image(img_path, output_path):
    threshold = 3
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    index = np.argwhere(img_gray > threshold)
    # print(index)
    hmin = index[:, 0].min()
    hmax = index[:, 0].max()
    # print(hmin, hmax)
    wmin = index[:, 1].min()
    wmax = index[:, 1].max()
    # print(wmin, wmax)
    output_file = os.path.join(output_path, img_path.split('/')[-1])
    cv2.imwrite(output_file, img[hmin:hmax, wmin:wmax])


if __name__ == '__main__':
    # test_img_root = '/mnt/d/MyDataBase/ODIR-5K/ODIR-5K_Testing_Images'
    # test_output_path = '../data/Test_Dataset'
    # test_images = os.listdir(test_img_root)
    # with tqdm(total=len(test_images)) as pbar:
    #     for img in test_images:
    #         img_path = os.path.join(test_img_root, img)
    #         deal_a_image(img_path, test_output_path)
    #         pbar.update(1)

    train_img_root = '/mnt/d/MyDataBase/ODIR-5K/ODIR-5K_Training_Dataset'
    train_output_path = '../data/Train_Dataset'
    train_images = os.listdir(train_img_root)
    with tqdm(total=len(train_images)) as pbar:
        for img in train_images:
            img_path = os.path.join(train_img_root, img)
            deal_a_image(img_path, train_output_path)
            pbar.update(1)
