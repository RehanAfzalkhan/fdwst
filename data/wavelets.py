import os
import numpy as np
import matplotlib.pyplot as plt
import pywt
import cv2
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import seaborn as sns

wavelet = 'haar'
def wavelets_of_images(blurry, sharp):
    cb = pywt.dwt2(blurry, wavelet)
    cs = pywt.dwt2(sharp, wavelet)
    return cb, cs

def inverse_wavelets_of_images(b, s):
    LLb, LHb, HLb, HHb = b
    LLs, LHs, HLs, HHs = s
    b = pywt.idwt2((LLb, (LHb, HLb, HHb)), wavelet)
    s = pywt.idwt2((LLs, (LHs, HLs, HHs)), wavelet)
    return b, s

def get_wavelets(blurry, sharp, name_A, name_B):

    levels     = 3
    show_combo = False
    save_img   = False

    name_A, name_B = name_A[:-4], name_B[:-4]
    B_img = b = (np.float32(np.array(Image.fromarray((np.array(blurry).transpose(1, 2, 0) * 255).astype(np.uint8)).convert("L"))) / 255)
    S_img = s = (np.float32(np.array(Image.fromarray((np.array(sharp ).transpose(1, 2, 0) * 255).astype(np.uint8)).convert("L"))) / 255)

    if show_combo:
        fig = plt.figure(figsize = (15, 9), layout = "tight")
        ax = fig.add_subplot(2, 2, 1)
        ax.imshow(b, cmap = "gray") # Blurry image
        ax.set_title("BLURRY")
        ax = fig.add_subplot(2, 2, 2)
        ax.imshow(s, cmap = "gray") # sharp image, this one is transferred to the blurry image
        ax.set_title("SHARP; TRANSFER THIS TO BLURRY")

    b_arr, s_arr = [], []
    temp_B, temp_S = [b], [s]

    # Forward Wavelet
    for _ in range(levels):

        b_arr, s_arr = temp_B, temp_S
        temp_B, temp_S = [], []

        for b, s in zip(b_arr, s_arr):
            (LLb, (LHb, HLb, HHb)), (LLs, (LHs, HLs, HHs)) = wavelets_of_images(b, s)
            temp_B.extend([LLb, LHb, HLb, HHb])
            temp_S.extend([LLs, LHs, HLs, HHs])

    B_waves = temp_B

    # Style Transfer
    c, top = 0, 2
    std_diffs = []
    l = len(temp_B) / 4

    for b, s in zip(temp_B, temp_S):
        if   c < l  : std_diffs.append(0)
        elif c >= l : std_diffs.append(abs(abs(b.std()) - abs(s.std())))
        std_diffs.append(abs(abs(b.std()) - abs(s.std())))
        c += 1

    ind = np.argpartition(std_diffs, -(top))[-(top):]

    c = 0
    B, S = [], []
    for b, s in zip(temp_B, temp_S):
        # Every Wavelet:    True
        # Top Left:         c >= l
        # Final LL filters: c % 4 != 0
        # Highest Var Diff: c in ind
        if True:
            b = (b - b.mean(axis = None)) / (b.std(axis = None) + 0.00001)
            b = (b * s.std(axis = None))  + s.mean(axis = None)
        B.append(b)
        S.append(s)
        c += 1

    temp_B, temp_S = B, S
    T_waves = temp_B

    # Inverse Wavelet
    for _ in range(levels):

        b_arr, s_arr = temp_B, temp_S
        temp_B, temp_S = [], []
        start, end, step = 0, len(b_arr), 4

        for j in range(start, end, step): 
            b, s = inverse_wavelets_of_images(b_arr[j : j + step], s_arr[j : j + step])
            temp_B.append(b)
            temp_S.append(s)

    T_img = temp_B[0]

    if show_combo:
        ax = fig.add_subplot(2, 2, 4)
        ax.imshow(T_img, cmap = "gray")
        ax.set_title("COMBO")
        plt.show()

    for T, B in zip(T_waves, B_waves):
            T = (T - T.min())/(T.max() - T.min() + 0.00001)
            B = (B - B.min())/(B.max() - B.min() + 0.00001)

    transform = transforms.Compose([transforms.ToTensor()])
    T_waves = transform(np.squeeze(np.asarray(T_waves, dtype = "float32")))
    B_waves = transform(np.squeeze(np.asarray(B_waves, dtype = "float32")))

    if save_img:
        T_img = (T_img - T_img.min())/(T_img.max() - T_img.min())
        img = Image.fromarray((T_img * 255.0).astype("uint8"))
        img.save(f"/home/admin/AdaAttN/datasets/wave_{levels}.png")

    return T_waves, B_waves, T_img, B_img