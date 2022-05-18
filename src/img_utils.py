import os
from PIL import Image, ImageOps


def rgb2gray(rgb_img_dpath, gray_img_dpath):
    for fname in os.listdir(rgb_img_dpath):
        if fname.endswith(".jpg"):
            print(fname)
            rgb_img = Image.open(os.path.join(rgb_img_dpath, fname))
            gray_img = ImageOps.grayscale(rgb_img)
            gray_img.save(os.path.join(gray_img_dpath, fname))


if __name__ == "__main__":
    rgb_img_dpath = os.path.join("multiobj/images")
    gray_img_dpath = os.path.join("multiobj/gray_images")
    rgb2gray(rgb_img_dpath, gray_img_dpath)
