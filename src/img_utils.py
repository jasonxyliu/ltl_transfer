import os
from PIL import Image, ImageOps


def rgb2gray(rgb_img_dpath, gray_img_dpath):
    for fname in os.listdir(rgb_img_dpath):
        if fname.endswith(".jpg"):
            print(fname)
            rgb_img = Image.open(os.path.join(rgb_img_dpath, fname))
            gray_img = ImageOps.grayscale(rgb_img)
            gray_img.save(os.path.join(gray_img_dpath, fname))


def rename(img_dpath, prefix):
    for fname in os.listdir(img_dpath):
        if fname.endswith(".jpg"):
            print(fname)
            img = Image.open(os.path.join(img_dpath, fname))
            new_fname = prefix + fname
            img.save(os.path.join(img_dpath, new_fname))


if __name__ == "__main__":
    # rgb2gray(os.path.join("multiobj/images"), os.path.join("multiobj/gray_images"))

    rename(os.path.join("multiobj/images_book_pr_hand_color"), "book_pr_")
