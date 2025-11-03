import os.path, cv2
from glob import glob
import matplotlib.pyplot as plt
import numpy as np

from_p = "/home/admin/AdaAttN/datasets/train_ridge_biocop_combo_sharp_above_400/"
to_p = "/home/admin/AdaAttN/datasets/train_ridge_biocop_combo_sharp_above_500/"
ext = ".png"	
c, t = 0, 500
max = 0

for imgp in sorted(glob(from_p + '*' + ext)):

	img = cv2.imread(imgp)
	# img = cv2.imread("/home/admin/AdaAttN/datasets/train_ridge_biocop_combo_sharp_above_250/2638860_05212014_3_3_ring_right.png")
	# img = cv2.resize(img, (256, 256))
	fm = cv2.Laplacian(img, cv2.CV_64F).var()

	if fm > max : 
		max = fm
		print(max)
		print(f'{os.path.basename(imgp)}: {fm}')

	# print(f'{os.path.basename(imgp)}: {fm}')

	# if fm > t:
		# plt.imshow(img)
		# plt.suptitle(fm)
		# plt.show()
		# cv2.imwrite(f"{to_p}{os.path.basename(imgp)}", cv2.imread(imgp)) 
		# c += 1

print(f"Total Images: {c}")
print(max)