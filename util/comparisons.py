import cv2
import os
import itertools
from skimage.metrics import structural_similarity
import numpy as np
from PIL import Image

s, cs = [], []

def compare(folder, experiment):

    psnr, ssim, mse = 0.0, 0.0, 0.0

    # for image in sorted(os.listdir(folder)):
        # ✅ Windows-safe folder check
    # folder = folder.replace('/home/admin/AdaAttN', os.getcwd())
        # ✅ Fix for correct Windows output directory
    folder = folder.replace('/home/admin/AdaAttN', os.getcwd())
    folder = folder.replace('/results/', '/outputs/')  # Fix path mismatch

    if not os.path.exists(folder):
        print(f"⚠️ Folder not found: {folder}")
        return
    print(f"🔍 Comparing images in: {folder}")
    for image in sorted(os.listdir(folder)):

        # ends
    
        if image.endswith("_s.png"):
            s.append(image)
        if image.endswith("_cs.png"):
            cs.append(image)

    for style, combo in zip(s, cs):
        img1 = cv2.imread(folder + "/" + style)
        img2 = cv2.imread(folder + "/" + combo)
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        (score, diff) = structural_similarity(img1_gray, img2_gray, full=True)
        h, w = img1_gray.shape
        dif = cv2.subtract(img1_gray, img2_gray)
        err = np.sum(dif**2)

        # Mask and difference code

        # diff = (diff * 255).astype("uint8")
        # thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours = contours[0] if len(contours) == 2 else contours[1]
        # mask = np.zeros(img1.shape, dtype='uint8')
        # filled_after = img2.copy()
        # for c in contours:
        #     area = cv2.contourArea(c)
        #     if area > 40:
        #         x,y,w,h = cv2.boundingRect(c)
        #         cv2.rectangle(img1, (x, y), (x + w, y + h), (36,255,12), 2)
        #         cv2.rectangle(img2, (x, y), (x + w, y + h), (36,255,12), 2)
        #         cv2.drawContours(mask, [c], 0, (0,255,0), -1)
        #         cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)
        # cv2.imshow('img1', img1)
        # cv2.imshow('img2', img2)
        # cv2.imshow('diff',diff)
        # cv2.imshow('mask',mask)
        # cv2.imshow('filled after',filled_after)
        # cv2.waitKey(0)

        psnr += cv2.PSNR(img1, img2)
        ssim += score
        mse += err/(float(h*w))
    psnr /= len(s)
    ssim /= len(s)
    mse /= len(s)
    print("%s Results:" % experiment)
    print("PSNR: %0.2f" % psnr)
    print("SSIM: %0.2f" % ssim)
    print("MSE: %0.2f" % mse)
