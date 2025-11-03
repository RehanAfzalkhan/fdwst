"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from os import listdir
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from util import comparisons
import webbrowser
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import shutil
import cv2
from PIL import Image

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)     
    shutil.rmtree('results/%s/test_latest/images' % opt.name, ignore_errors=True)          # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        # save_images(webpage, visuals, img_path, width=opt.display_winsize)
        # ✅ Ensure img_path is always valid starts replace code 
        if isinstance(img_path, list) and len(img_path) > 0:
            valid_path = img_path[0]
        else:
            valid_path = "data/content_blurry/1.png"  # fallback to first image

        save_images(webpage, visuals, [valid_path], width=opt.display_winsize)

        # replace code ends 

    webpage.save()  # save the HTML
    # comparisons.compare('/home/admin/AdaAttN/results/%s/test_latest/images' % opt.name, opt.name)
    # webbrowser.open_new_tab('/home/admin/AdaAttN/results/%s/test_latest/index.html' % opt.name)
    # ✅ Updated to match Windows path and "outputs" directory
    output_dir = os.path.join(os.getcwd(), "outputs", opt.name, "test_latest")
    image_dir = os.path.join(output_dir, "images")
    index_html = os.path.join(output_dir, "index.html")

    comparisons.compare(image_dir, opt.name)

    # ✅ Automatically open correct index.html in browser
    if os.path.exists(index_html):
        print(f"📂 Opening results viewer at: {index_html}")
        webbrowser.open(f"file:///{index_html}")
    else:
        print(f"⚠️ index.html not found in: {index_html}")


# -----------------------------------------------------------
# 📊 Post-Evaluation: Save Results (PSNR, SSIM, MSE) to CSV
# -----------------------------------------------------------
import os
import csv
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from PIL import Image
import glob

# Path to folder containing result images
images_dir = os.path.join(output_dir, "images")
csv_path = os.path.join(output_dir, "fdwst_results.csv")

if not os.path.exists(images_dir):
    print(f"⚠️ No output images found in {images_dir}")
else:
    print(f"📁 Calculating metrics for all images in: {images_dir}")

    blurry_dir = os.path.join(opt.content_path)
    sharp_dir = os.path.join(opt.style_path)

    blurry_images = sorted(glob.glob(os.path.join(blurry_dir, "*.png")))
    sharp_images = sorted(glob.glob(os.path.join(sharp_dir, "*.png")))
    result_images = sorted(glob.glob(os.path.join(images_dir, "*.png")))

    psnr_list, ssim_list, mse_list, names = [], [], [], []

    for i, result_path in enumerate(result_images):
        # Match image names
        img_name = os.path.basename(result_path)
        blur_path = os.path.join(blurry_dir, img_name)
        sharp_path = os.path.join(sharp_dir, img_name)

        if not (os.path.exists(blur_path) and os.path.exists(sharp_path)):
            continue

        try:
            img_blur = np.array(Image.open(blur_path).convert("RGB"))
            img_sharp = np.array(Image.open(sharp_path).convert("RGB"))
            img_result = np.array(Image.open(result_path).convert("RGB"))

            # Compute metrics
            psnr_val = psnr(img_sharp, img_result, data_range=255)
            ssim_val = ssim(img_sharp, img_result, channel_axis=2, data_range=255)
            mse_val = mse(img_sharp, img_result)

            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            mse_list.append(mse_val)
            names.append(img_name)

        except Exception as e:
            print(f"⚠️ Error processing {img_name}: {e}")

    # Save to CSV
    df = pd.DataFrame({
        "Image_Name": names,
        "PSNR": psnr_list,
        "SSIM": ssim_list,
        "MSE": mse_list
    })
    df.to_csv(csv_path, index=False)

    # Summary metrics
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_mse = np.mean(mse_list)

    print("\n✅ Evaluation Complete!")
    print(f"📊 Average PSNR: {avg_psnr:.2f}")
    print(f"📊 Average SSIM: {avg_ssim:.2f}")
    print(f"📊 Average MSE: {avg_mse:.2f}")
    print(f"📁 Detailed results saved in: {csv_path}")



