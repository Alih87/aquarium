import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from PIL import Image
from skimage import exposure
import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument("imgs")
parser.add_argument("labels")
parser.add_argument("save")

def identify_correct(IMG_DIR, LBL_DIR):
    lbls = os.listdir(LBL_DIR)
    lbls_complete_pths = [LBL_DIR + pth for pth in lbls]

    img_pths = []
    lbl_pths = []

    for pth in lbls_complete_pths:
        with open(pth, 'r+') as f:
            try:
                content = f.readlines()[0]
            except IndexError:
                print("[INFO] No label")
                f.write("7")
        f.close()
        img_pths.append(IMG_DIR+pth.split("/")[-1][:-3]+"jpg")
        lbl_pths.append(pth)
    
    assert len(img_pths) == len(lbl_pths)

    return img_pths, lbl_pths

def save_data(SAVE_DIR, IMG_DIRS, LBL_DIRS, equalize=False):
    count = 0
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)
        os.mkdir(SAVE_DIR+"images/")
        os.mkdir(SAVE_DIR+"labels/")
    else:
        for i in os.listdir(SAVE_DIR+"images/"):
            os.remove(SAVE_DIR+"images/"+i)
        for i in os.listdir(SAVE_DIR+"labels/"):
            os.remove(SAVE_DIR+"labels/"+i)

    for ipth, lpth in zip(IMG_DIRS,LBL_DIRS):
        vertices = []
        img = (resize((imread(ipth, as_gray=True)), (512, 320))*255).astype(np.uint8)
        if equalize:
            eq_img = exposure.equalize_adapthist(img, clip_limit=0.03)
        else:
            eq_img = img
        eq_img = (eq_img * 255).astype(np.uint8)
        with open(lpth, 'r') as f:
            content = f.readlines()[0].split(" ")
        f.close()
        if len(content) > 1:
            content = list(map(float, content))
            content[0] = int(content[0]+1)
            coords = content[1:]
            for i in range(0, len(coords)-2, 2):
                vertices.append((coords[i]*img.shape[1], coords[i+1]*img.shape[0]))
            vertices = np.asanyarray(vertices)
            x_min = np.amin(vertices[:,0])
            x_max = np.amax(vertices[:,0])
            y_min = np.amin(vertices[:,1])
            y_max = np.amax(vertices[:,1])
            with open(SAVE_DIR+f"labels/{count}.txt", 'w') as f:
                for v in [x_min, y_min, x_max, y_max]:
                    f.write(str(v)+" ")
                f.write(str(content[0]))
            
            Image.fromarray(eq_img).save(SAVE_DIR+f"images/{count}.jpg")
            count += 1
        else:
            continue

        
if __name__ == '__main__':
    IMG_DIR = r"/data_hdd1/hassan/projects/fracture/data/test/images/"
    LBL_DIR = r"/data_hdd1/hassan/projects/fracture/data/test/labels/"
    SAVE_DIR = r"/data_hdd1/hassan/projects/fracture/data/test/clean/"

    img_pths, lbl_pths = identify_correct(IMG_DIR, LBL_DIR)
    save_data(SAVE_DIR, img_pths, lbl_pths, equalize=True)

