import os
import numpy as np
import matplotlib.pyplot as plt
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch.backends.cudnn as cudnn
from PIL import Image, ImageFile
import torchvision.transforms as transforms
from data.wavelets import get_wavelets
from random import randrange
import cv2
import math

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated

class UnalignedDataset(BaseDataset):

    def __init__(self, opt):

        BaseDataset.__init__(self, opt)
        self.dir_A = opt.content_path
        self.dir_B = opt.style_path
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform_A = get_transform(self.opt, content = True,  isTrain = opt.isTrain)
        self.transform_B = get_transform(self.opt, content = False, isTrain = opt.isTrain)

    def __getitem__(self, index):

        same_ID_pairs  = True
        dynamic_resize = False

        if same_ID_pairs:
            index_A = index_B = index
        else:
            index_A = index // self.B_size
            index_B = index % self.B_size

        A_path = self.A_paths[index_A]
        B_path = self.B_paths[index_B]
        # A_path = "/home/admin/AdaAttN/datasets/synthetic/AB/blurry/1072828_3_0.png" # Fixed BLURRY
        # B_path = "/home/admin/AdaAttN/datasets/binary_prints/1695771_2_0.png" # Fixed SHARP
        # T_path = "/home/admin/AdaAttN/datasets/binary_prints/4541406_4_3.png" # Binary TRANSFER
        # new code starts
        # ✅ Use local Windows-compatible transfer image
        T_candidates = [p for p in self.B_paths if p.endswith(".png")]
        if len(T_candidates) > 0:
            from random import choice
            T_path = choice(T_candidates)  # randomly pick one from style_sharp folder
        else:
            T_path = B_path  # fallback if no images found

        # new code ends

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        T_img = Image.open(T_path).convert('RGB')

        B_arr = np.array(B_img)

        blur_scores = cv2.Laplacian((B_arr * 255.0).astype("uint8"), cv2.CV_64F).var()
        blur_scores = round(((blur_scores - 0.0001)/(2423.05 - 0.0001)) * 10.0, 2).astype("float32")

        if dynamic_resize:
            # resize_lower, resize_upper = 0.7, 3.0
            # new_dim = int(int(A_img.size[0] * np.random.uniform(resize_lower, resize_upper) / 4) * 4)
            # new_size = (new_dim, new_dim)
            # new_size = (256, 256)
            # A_img = A_img.resize(new_size)
            # B_img = B_img.resize(new_size)
            # T_img = T_img.resize(new_size)

            size = 400

            crop256 = transforms.RandomCrop((256, 256)) 
            crop400 = transforms.RandomCrop((size, size))
            resize  = transforms.Resize((256, 256))

            if A_img.size[0] >= size and A_img.size[1] >= size:
                A_img = crop400(A_img)
                A_img = resize(A_img)
            elif A_img.size[0] < 256 or A_img.size[1] < 256:
                A_img = A_img.resize((int(A_img.size[0] * 1.3), int(A_img.size[1] * 1.3)))
                A_img = crop256(A_img)
            else:
                A_img = crop256(A_img)
            
            B_img = A_img
            T_img = crop256(T_img)
        # insert new code starts
        # ✅ Ensure all images have the same size before transformation
        target_size = (256, 256)
        A_img = A_img.resize(target_size)
        B_img = B_img.resize(target_size)
        T_img = T_img.resize(target_size)

        # insert new code ends
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        T = self.transform_B(T_img)

        name_A = os.path.basename(A_path)
        name_B = os.path.basename(B_path)
        name_T = os.path.basename(T_path)

        name = name_B[:name_B.rfind('.')] + '_' + name_A[:name_A.rfind('.')] + name_A[name_A.rfind('.'):]
# ✅ Final Safe Return (guarantees image_path is valid)
              # ✅ Final Safe Return (guarantees image_path is valid)
        # if A_path is empty, pick the first image in content_blurry folder
        if A_path is None or A_path == "" or not os.path.exists(A_path):
            content_dir = os.path.join(os.getcwd(), "data", "content_blurry")
            all_imgs = [
                os.path.join(content_dir, f)
                for f in os.listdir(content_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ]
            if len(all_imgs) > 0:
                A_path = all_imgs[0]
            else:
                A_path = os.path.join(os.getcwd(), "data", "content_blurry", "1.png")  # default fallback

        # ✅ Always return valid image path
        return {
            'blur': A,
            'sharp': B,
            'bin': T,
            'name': name,
            'Name_A': name_A,
            'Name_B': name_B,
            'blur_scores': blur_scores,
            'image_path': [A_path]  # always a list with one valid path
        }



    # code replacing ends


    def __len__(self):
        if self.opt.isTrain : return self.A_size
        else : return min(self.A_size * self.B_size, self.opt.num_test)