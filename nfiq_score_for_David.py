import glob
import math
import os
import random
import subprocess
import multiprocessing as mp
from p_tqdm import p_imap
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import wsq
from tqdm import tqdm
import imutils
import seaborn as sns
import pathlib
from scipy.stats import norm
random.seed(1)

def tiffer(source, domain, padding=False):
	img_dir_list = glob.glob(source)
	print('Converting {} images to TIFF file.\n'.format(len(img_dir_list)))
	target_path = f'/home/admin/Documents/temp/tiff/{domain}'
	pathlib.Path(target_path).mkdir(exist_ok=True, parents=True)
	for image in img_dir_list:
		img_name = os.path.splitext(os.path.basename(image))[0]
		img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
		if padding:
			h, w = img.shape
			max_wh = max(h, w)
			hp = int((max_wh - w) / 2)
			vp = int((max_wh - h) / 2)
			img = cv2.copyMakeBorder(img, left=hp, top=vp, right=hp, bottom=vp, borderType=cv2.BORDER_CONSTANT, value=0)
			img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
		c_img = img

		cv2.imwrite(f"{target_path}/{img_name}.tiff", c_img)



def wsqer(source, source_type):
	print('Converting {} TIFF to WSQ\n'.format(source_type))
	target_path = f'/home/admin/Documents/temp/wsq/{source_type}'
	pathlib.Path(target_path).mkdir(exist_ok=True, parents=True)
	for img in tqdm(glob.glob(source)):
		image = Image.open(img)
		img_name = os.path.splitext(os.path.basename(img))[0]
		image.save(f'{target_path}/{img_name}.wsq')

def parallel_scorer(image):

	img_name = os.path.basename(image)
	p = subprocess.Popen(["/usr/local/nfiq2/bin/nfiq2", "{}".format(image)], stdout=subprocess.PIPE,
						 universal_newlines=True)
	output, err = p.communicate()
	# output = output.split(' ')[0]
	# print(i, output)
	# f.write("{0} {1}".format(img_name, output))

	if len(output) > 5:
		# f.write("{0} {1}\n".format(img_name, '0'))
		return (img_name, '0\n')
	else:
		# f.write("{0} {1}".format(img_name, output))
		return (img_name, output)


def scorer2(wsq_path, domain, file_path='/home/n-lab/Amol/quality', img_ext='png'):
	f = open(os.path.join(file_path, f'nfiq_biocop_before_kept.txt'), 'w')
	img_dir_list = glob.glob(wsq_path)
	print('Calculating scores of {0} {1} wsq images.\n'.format(len(img_dir_list), domain))
	pool = mp.Pool(mp.cpu_count())
	for score in p_imap(parallel_scorer, [image for image in img_dir_list]):
		# print(f"{score[0]} {score[1]}")
		f.write(f"{score[0].replace('wsq', img_ext)} {score[1]}")
	pool.close()
	f.close()



domain = 'biocop_before_kept'
tiffer(f'/home/admin/AdaAttN/datasets/biocop_potatoes_removed/original_biocop/kept/*', domain, padding=True)
wsqer(f'/home/admin/Documents/temp/tiff/{domain}/*.tiff', domain)
scorer2(f'/home/admin/Documents/temp/wsq/{domain}/*.wsq', domain,
		file_path='/home/admin/Documents/temp', img_ext='png')


