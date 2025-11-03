import os

p = "/home/admin/AdaAttN/datasets/deblurring_fingerphoto_dataset/train"

curr = ""
c = 0

for im in sorted(os.listdir(p)):
    # if "_s." in im or "_blur." in im or "_c." in im: 
    #     os.remove(os.path.join(p, im))
    #     continue

    # IITB
    # r = im.find("_c")
    # strg = im[:r] + '.png'

    # Ridgebase
    # r = im.find("_1_A")
    # if r == -1 : r = im.find("_1_g")
    # strg = im[:r] + ".png"

    strg = im[:7]

    if curr != strg:
        curr = strg 
        c += 1

    print(c)
    
    # Biocop Real
    # i = im.find('t_1') 
    # strg = im[:i] + "t.png"

    # Final Rename
    # print(strg)
    # os.rename(f"{p}/{im}", f"{p}/{strg}")