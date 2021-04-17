import os

import cv2
import numpy as np
import tensorflow as tf
from imgaug import augmenters as iaa

import augmentationMethods as am
import config


def cropImg(img, size):
    width, height = img.shape[1], img.shape[0]
    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(size / 2), int(size / 2)
    crop_img = img[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]
    return crop_img


def removeNumberAndExt(img: str):
    return img[:-7]


def getUniques(labels):
    used = set()
    unique = [x for x in labels if x not in used and (used.add(x) or True)]
    dict = {}
    for i in range(0, len(unique)):
        dict[unique[i]] = np.uint8(i)
    return unique, dict


def double_labels(labs, count: int = 2):
    list = []
    for i in labs:
        for j in range(0, count):
            list.append(i)
    return list


def convertLabel(lijst):
    ans = []
    classes = ['handShake', 'highFive', 'hug', 'kiss']
    for l in lijst:
        ans.append(classes.index(l))
    return np.array(ans)


def take_middle_frame(video_folder):
    for video_file in os.listdir(video_folder):
        video_path = config.TV_VIDEOS_SLASH + str(video_file)
        cap = cv2.VideoCapture(video_path)
        total_frames = cap.get(7)  # magic for taking prop-Id
        cap.set(1, int(total_frames / 2))
        ret, frame = cap.read()
        cv2.imwrite(config.TV_IMG + video_file[:-4] + ".jpg", frame)


def splitTrain(files, labels):
    if len(files) != len(labels):
        print("ERROR! Files and labels don't have the same length")
        return
    train_imgs, val_imgs, train_labels, val_labels = [], [], [], []
    temp = []
    prev_lab = removeNumberAndExt(files[0])
    files.append("END_000.jpg")
    for i in range(0, len(files)):
        lab = removeNumberAndExt(files[i])
        if lab != prev_lab:
            aantalTrain = len(temp) - int(len(temp) * config.Validate_perc)
            # print(f"Length for lab: {len(temp)} and number in Train: {aantalTrain}")
            for j in temp[:aantalTrain]:
                train_imgs.append(files[j])
                train_labels.append(labels[j])
            for j in temp[aantalTrain:]:
                val_imgs.append(files[j])
                val_labels.append(labels[j])

            # for new label
            prev_lab = lab
            temp = [i]
        else:
            temp.append(i)
    return train_imgs, np.array(train_labels), val_imgs, np.array(val_labels)


def readImgs(imgs, location: str, grayScale: bool):
    lijst = []
    gray = 0 if grayScale else 1
    for fileName in imgs:
        img = cv2.imread(location + fileName, gray)
        if img is None:
            print(f"None! {fileName}")
        else:
            if grayScale:
                img = img.reshape((config.Image_size, config.Image_size, 1))
            lijst.append(img)
    return lijst


def augmentImages(imgs, lowHue: bool, highSatur: bool, lowSatur: bool, highBright: bool,
                  lowBright: bool, flip: bool, addInvert: bool, txRight: bool = True, txLeft: bool = True, blurImg: bool = True):
    blur = iaa.GaussianBlur(sigma=(0.9, 1.0)).to_deterministic()
    tx_rechts = iaa.TranslateX(px=(19, 20), mode="reflect").to_deterministic()
    tx_links = iaa.TranslateX(px=(-19, -20), mode="reflect").to_deterministic()

    lijst = []
    count = 1
    for i in [flip, addInvert, highBright, lowBright, lowSatur, highSatur, lowHue, txLeft, txRight, blurImg]:
        if i: count += 1
    for img in imgs:
        lijst.append(img)  # add original
        if blurImg:
            blur_img = blur(image=img)
            lijst.append(blur_img)
        if txRight:
            tx_rechts_img = tx_rechts(image=img)
            lijst.append(tx_rechts_img)
        if txLeft:
            tx_links_img = tx_links(image=img)
            lijst.append(tx_links_img)
        # if addGray:
        #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #     lijst.append(gray)
        if addInvert:
            invert = (255 - img)
            lijst.append(invert)
        if flip:
            flipped = cv2.flip(img, 1)
            lijst.append(flipped)
        if highBright:
            hb = tf.image.adjust_brightness(img, am.high_bright)
            lijst.append(hb.numpy())
        if lowBright:
            lb = tf.image.adjust_brightness(img, am.low_bright)
            lijst.append(lb.numpy())
        if highSatur:
            hs = tf.image.adjust_saturation(img, am.high_satur)
            lijst.append(hs.numpy())
        if lowSatur:
            ls = tf.image.adjust_saturation(img, am.low_satur)
            lijst.append(ls.numpy())
        # if highHue:
        #     hh = tf.image.adjust_hue(img, am.high_hue)
        #     lijst.append(hh.numpy())
        if lowHue:
            lh = tf.image.adjust_hue(img, am.low_hue)
            lijst.append(lh.numpy())
    return lijst, count

# imgs = cv2.imread("J:\\Python computer vision\\Action_recognition-CV\\Data\\Stanford40\\ImagesConvCrop\\applauding_001.jpg",1)
# cv2.imshow('img', imgs)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# p = tf.keras.preprocessing.image.random_shift(imgs, 0.2, 0.0)
# p = tf.keras.preprocessing.image.random_shear(imgs, 20)
# p = tf.image.adjust_saturation(imgs, 0.5).numpy()
# p = tf.image.adjust_hue(imgs, 0.1).numpy()
# augmImgs, count = augmentImages([imgs], True, False, False, False, False, False, False)
# p = tfa.image.translate(imgs, tf.constant([0.5, 0])).numpy()
# cv2.imshow('img', p)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
