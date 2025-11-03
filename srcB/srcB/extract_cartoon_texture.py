import subprocess
from glob import glob
import os
import pathlib
from tqdm import tqdm
from multiprocessing import Pool

from PIL import Image#Grab, Image
import io

source_dir = glob('/home/admin/AdaAttN/datasets/synthetic/AB/blurry/*.png')

target_dir = '/home/admin/AdaAttN/datasets/synthetic_texture'
cartoon_target = '/home/admin/AdaAttN/datasets/synthetic_cartoon'
pathlib.Path(target_dir).mkdir(exist_ok=True, parents=True)
pathlib.Path(cartoon_target).mkdir(exist_ok=True, parents=True)

def run_subprocess(img):
    name = os.path.basename(img)
    p = subprocess.Popen(["/home/admin/AdaAttN/srcB/srcB/cartoonIpol", img, "6",
                          f"{cartoon_target}/{name}",
                          f"{os.path.join(target_dir, name)}"], stdout=subprocess.PIPE, universal_newlines=True)
    # output, err = p.communicate()
    p.wait()


if __name__ == '__main__':

    # Number of processes to run in parallel
    num_processes = 3

    # Create a pool of processes
    pool = Pool(processes=num_processes)

    # Map the subprocess commands to the pool of processes
    # pool.map(run_subprocess, source_dir)

    # Initialize the progress bar
    progress_bar = tqdm(total=len(source_dir), desc='Running subprocesses')

    # Map the subprocess commands to the pool of processes
    for _ in pool.imap_unordered(run_subprocess, source_dir):
        # Update the progress bar
        progress_bar.update()

    # Close the pool and wait for the processes to finish
    pool.close()
    pool.join()

    # Close the progress bar
    progress_bar.close()
