import os
import numpy as np
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image, ImageFile
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import cv2
from random import choice

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


class UnalignedDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = opt.content_path
        self.dir_B = opt.style_path
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform_A = get_transform(self.opt, content=True, isTrain=opt.isTrain)
        self.transform_B = get_transform(self.opt, content=False, isTrain=opt.isTrain)

    def __getitem__(self, index):
        # --- Pair indices ---
        index_A = index % self.A_size
        index_B = index % self.B_size

        A_path = self.A_paths[index_A]
        B_path = self.B_paths[index_B]

        # --- Safe fallback for T_path ---
        T_path = choice(self.B_paths) if len(self.B_paths) > 0 else B_path

        # --- Load and preprocess ---
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        T_img = Image.open(T_path).convert('RGB')

        # Resize to match
        target_size = (256, 256)
        A_img = A_img.resize(target_size)
        B_img = B_img.resize(target_size)
        T_img = T_img.resize(target_size)

        # Apply transforms
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        T = self.transform_B(T_img)

        # Blur score
        B_arr = np.array(B_img)
        blur_scores = cv2.Laplacian((B_arr * 255.0).astype("uint8"), cv2.CV_64F).var()
        blur_scores = round(((blur_scores - 0.0001) / (2423.05 - 0.0001)) * 10.0, 2)

        # --- File names ---
        name_A = os.path.basename(A_path)
        name_B = os.path.basename(B_path)
        name = f"{os.path.splitext(name_B)[0]}_{os.path.basename(name_A)}"

        # --- Guarantee a valid image path ---
        if A_path is None or not os.path.exists(A_path):
            content_dir = os.path.join(os.getcwd(), "data", "content_blurry")
            all_imgs = [
                os.path.join(content_dir, f)
                for f in os.listdir(content_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ]
            A_path = all_imgs[0] if len(all_imgs) > 0 else "data/content_blurry/1.png"

        # --- Final Safe Return ---
        if A_path is None or not os.path.exists(A_path):
            A_path = "data/content_blurry/1.png"
        if not isinstance(A_path, list):
            A_path = [A_path]
        return {
                'blur': A,
                'sharp': B,
                'bin': T,
                'name': name,
                'Name_A': name_A,
                'Name_B': name_B,
                'blur_scores': blur_scores,
                'image_path': A_path  # always a list
            }

    def __len__(self):
        return min(self.A_size, self.opt.num_test)
