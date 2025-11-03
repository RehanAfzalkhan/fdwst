import os
from PIL import Image

folders = ["data/content_blurry", "data/style_sharp"]

for folder in folders:
    for filename in os.listdir(folder):
        if filename.lower().endswith(".bmp"):
            bmp_path = os.path.join(folder, filename)
            png_path = os.path.splitext(bmp_path)[0] + ".png"
            Image.open(bmp_path).save(png_path)
            print(f"Converted: {filename} → {os.path.basename(png_path)}")
