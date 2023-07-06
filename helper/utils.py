import os
import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml

import torch
from tqdm.auto import tqdm


def load_yaml(path):
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return config


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# RLE 디코딩 함수
def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def show_image(img, label):
    img = img.permute(1, 2, 0).numpy()

    plt.figure(figsize=(10, 10))

    plt.subplot(1, 2, 1)
    plt.xticks(range(0, img.shape[0], 50))
    plt.yticks(range(0, img.shape[0], 50))
    plt.title("Image", fontsize=15, pad=10)
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    plt.xticks(range(0, img.shape[0], 50))
    plt.yticks(range(0, img.shape[0], 50))
    plt.title("Mask", fontsize=15, pad=10)
    plt.imshow(label, cmap='gray')

    plt.show()


def make_parquet(path, shape):
    df = pd.read_csv(path)
    mask_rle = df['mask_rle'].copy().str.split()
    mask_rle = mask_rle.apply(lambda x: np.asarray(x, dtype=np.int32))

    i = 0
    for rle in tqdm(mask_rle, unit='imgs'):
        rle = rle.reshape(-1, 2)
        starts, lengths = rle[:, 0], rle[:, 1]
        starts -= 1
        ends = starts + lengths

        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        df['mask_rle'][i] = img.tobytes()
        i += 1


    print('Creating parquet file ... ')
    df.to_parquet('./train.parquet')
    print('Done.')
