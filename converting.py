import cv2

import config
import HelperFunctions as HF


def preprocess_stanf(stanford_x_imgs, save: bool = False, aug: bool = True):
    lijst = []
    for fileName in stanford_x_imgs:
        img = cv2.imread(config.STANF_IMG + fileName, 0)
        img = cv2.resize(img, (config.Image_size, config.Image_size))
        if aug and save:
            img_flip = cv2.flip(img, 1)
            cv2.imwrite(config.STANF_CONV + fileName[:-4] + "_flip.jpg", img_flip)
        if save:
            cv2.imwrite(config.STANF_CONV + fileName, img)
        if img is None:
            print(f"None! {fileName}")
        else:
            img = img.reshape((config.Image_size, config.Image_size, 1))
            lijst.append(img)
    return lijst


def preprocess_tv(tv_x_imgs, save: bool = False, aug: bool = True):
    list = []
    for filename in tv_x_imgs:
        img = cv2.imread(config.TV_IMG + filename, 0)
        img = cv2.resize(img, (config.Image_size, config.Image_size))
        if aug and save:
            img_flip = cv2.flip(img, 1)
            cv2.imwrite(config.STANF_CONV + filename[:-4] + "_flip.jpg", img_flip)
        if save:
            cv2.imwrite(config.STANF_CONV + filename, img)
        if img is None:
            print(f"None! {filename}")
        else:
            img = img.reshape((config.Image_size, config.Image_size, 1))
            list.append(img)
    return list


def convertAndCropImg(x_imgs, readLoc, crop: bool, resize: bool, size: int, saveLoc: str):
    for fileName in x_imgs:
        img = cv2.imread(readLoc + fileName, 0)
        if img is not None:
            if crop:
                a, b = img.shape[0], img.shape[1]
                img = HF.cropImg(img, min(a, b))
            if resize:
                img = cv2.resize(img, (size, size))

            cv2.imwrite(saveLoc + fileName, img)
            img_flip = cv2.flip(img, 1)
            cv2.imwrite(saveLoc + fileName[:-4] + "_flip.jpg", img_flip)
        else:
            print(f"None! {fileName}")
    print("Done converting!")


def convertNew(x_imgs, readLoc, size: int, saveLoc: str = "", saveCropLoc: str = "", grayScale: bool = False):
    gray = 0 if grayScale else 1
    for fileName in x_imgs:
        img = cv2.imread(readLoc + fileName, gray)
        if img is not None:
            if len(saveCropLoc) > 2:
                a, b = img.shape[0], img.shape[1]
                imgc = HF.cropImg(img, min(a, b))
                imgc = cv2.resize(imgc, (size, size))
                cv2.imwrite(saveCropLoc + fileName, imgc)

                img_flip = cv2.flip(imgc, 1)
                cv2.imwrite(saveCropLoc + fileName[:-4] + "_flip.jpg", img_flip)
            if len(saveLoc) > 2:
                imgr = cv2.resize(img, (size, size))
                cv2.imwrite(saveLoc + fileName, imgr)

                img_flip = cv2.flip(imgr, 1)
                cv2.imwrite(saveLoc + fileName[:-4] + "_flip.jpg", img_flip)
        else:
            print(f"None! {fileName}")
    print("Done converting!")
