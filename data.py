import cv2
import numpy as np
from sklearn.model_selection import train_test_split

import HelperFunctions as HF
import OpticalFlow as OptF
import config
import os


def train_test_stanford(printing: bool = False):
    with open('./Data/Stanford40/ImageSplits/train.txt', 'r') as f:
        train_files = list(map(str.strip, f.readlines()))
        train_labels = ['_'.join(name.split('_')[:-1]) for name in train_files]
        if printing:
            print(f'Train files ({len(train_files)}):\n\t{train_files}')
            print(f'Train labels ({len(train_labels)}):\n\t{train_labels}\n')

    with open('./Data/Stanford40/ImageSplits/test.txt', 'r') as f:
        test_files = list(map(str.strip, f.readlines()))
        test_labels = ['_'.join(name.split('_')[:-1]) for name in test_files]
        if printing:
            print(f'Test files ({len(test_files)}):\n\t{test_files}')
            print(f'Test labels ({len(test_labels)}):\n\t{test_labels}\n')

    action_categories = sorted(list(set(['_'.join(name.split('_')[:-1]) for name in train_files])))
    if printing:
        print(f'Action categories ({len(action_categories)}):\n{action_categories}')

    return train_files, train_labels, test_files, test_labels


def train_tests_tv(printing: bool = False):
    set_1_indices = [
        [2, 14, 15, 16, 18, 19, 20, 21, 24, 25, 26, 27, 28, 32, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
        [1, 6, 7, 8, 9, 10, 11, 12, 13, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 44, 45, 47, 48],
        [2, 3, 4, 11, 12, 15, 16, 17, 18, 20, 21, 27, 29, 30, 31, 32, 33, 34, 35, 36, 42, 44, 46, 49, 50],
        [1, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 22, 23, 24, 26, 29, 31, 35, 36, 38, 39, 40, 41, 42]]
    set_2_indices = [[1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 22, 23, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39],
                     [2, 3, 4, 5, 14, 15, 16, 17, 18, 19, 20, 21, 22, 26, 36, 37, 38, 39, 40, 41, 42, 43, 46, 49, 50],
                     [1, 5, 6, 7, 8, 9, 10, 13, 14, 19, 22, 23, 24, 25, 26, 28, 37, 38, 39, 40, 41, 43, 45, 47, 48],
                     [2, 3, 4, 5, 6, 15, 19, 20, 21, 25, 27, 28, 30, 32, 33, 34, 37, 43, 44, 45, 46, 47, 48, 49, 50]]
    classes = ['handShake', 'highFive', 'hug', 'kiss']  # we ignore the negative class

    # test set
    set_1 = [f'{classes[c]}_{i:04d}.avi' for c in range(len(classes)) for i in set_1_indices[c]]
    set_1_label = [f'{classes[c]}' for c in range(len(classes)) for i in set_1_indices[c]]
    if printing:
        print(f'Set 1 to be used for test ({len(set_1)}):\n\t{set_1}')
        print(f'Set 1 labels ({len(set_1_label)}):\n\t{set_1_label}\n')

    # training set
    set_2 = [f'{classes[c]}_{i:04d}.avi' for c in range(len(classes)) for i in set_2_indices[c]]
    set_2_label = [f'{classes[c]}' for c in range(len(classes)) for i in set_2_indices[c]]
    if printing:
        print(f'Set 2 to be used for train and validation ({len(set_2)}):\n\t{set_2}')
        print(f'Set 2 labels ({len(set_2_label)}):\n\t{set_2_label}')
    return set_1, set_1_label, set_2, set_2_label  # testx, testy, trainx, trainy


def getDataSet(files, location: str, grayScale: bool, labels, aug: bool = True, leave_one_out: bool = False,
               leave_out_ind: int = 1, fuse: bool = False):
    imgs = HF.readImgs(files, location, grayScale)
    if leave_one_out:
        if leave_out_ind == 1:
            augmImgs, count = HF.augmentImages(imgs, False, True, True, True, True, True, True, True, True, True)
        elif leave_out_ind == 2:
            augmImgs, count = HF.augmentImages(imgs, True, False, True, True, True, True, True, True, True, True)
        elif leave_out_ind == 3:
            augmImgs, count = HF.augmentImages(imgs, True, True, False, True, True, True, True, True, True, True)
        elif leave_out_ind == 4:
            augmImgs, count = HF.augmentImages(imgs, True, True, True, False, True, True, True, True, True, True)
        elif leave_out_ind == 5:
            augmImgs, count = HF.augmentImages(imgs, True, True, True, True, False, True, True, True, True, True)
        elif leave_out_ind == 6:
            augmImgs, count = HF.augmentImages(imgs, True, True, True, True, True, False, True, True, True, True)
        elif leave_out_ind == 7:
            augmImgs, count = HF.augmentImages(imgs, True, True, True, True, True, True, False, True, True, True)
        elif leave_out_ind == 8:
            augmImgs, count = HF.augmentImages(imgs, True, True, True, True, True, True, True, False, True, True)
        elif leave_out_ind == 9:
            augmImgs, count = HF.augmentImages(imgs, True, True, True, True, True, True, True, True, False, True)
        elif leave_out_ind == 10:
            augmImgs, count = HF.augmentImages(imgs, True, True, True, True, True, True, True, True, True, False)
        elif leave_out_ind == 11:
            augmImgs, count = HF.augmentImages(imgs, True, True, True, True, True, True, True, True, True, True)
        else:
            print("leave_out_ind invalid value, taking default augmentation")
            augmImgs, count = HF.augmentImages(imgs, False, True, True, True, True, True, True, True, True, True)
        newLabels = HF.double_labels(labels, count)
    elif aug and not leave_one_out:
        augmImgs, count = HF.augmentImages(imgs, True, True, True, True, True, True, True, True, True, True)
        newLabels = HF.double_labels(labels, count)
    elif fuse:
        augmImgs, count = HF.augmentImages(imgs, False, False, False, False, True, False, False, False, False, False)

        newLabels = HF.double_labels(labels, count)
    else:
        return np.array(imgs), np.array(labels)

    return np.array(augmImgs), np.array(newLabels)


def getStanfordData(leave_one_out: bool = False, leave_out_ind: int = 1):
    stf_train_files, stf_train_labels_S, stf_test_files, stf_test_labels = train_test_stanford(False)

    uniqueLabels, dictionary = HF.getUniques(stf_test_labels)
    stf_train_labels_ind = [dictionary[lab] for lab in stf_train_labels_S]
    stf_test_labels_ind = [dictionary[lab] for lab in stf_test_labels]

    imgs_train, labs_train = getDataSet(stf_train_files, config.STANF_CONV_CROP, False, stf_train_labels_ind, aug=True,
                                        leave_one_out=leave_one_out, leave_out_ind=leave_out_ind)
    imgs_test, labs_test = getDataSet(stf_test_files, config.STANF_CONV_CROP, False, stf_test_labels_ind, aug=False)
    # print(imgs_train.shape)
    # print(labs_train.shape)
    # print(imgs_test.shape)
    # print(labs_test.shape)
    return imgs_train, labs_train, imgs_test, labs_test




def loadFlowData(augm: bool, extra_data: bool, threeD: bool):
    opties = ["flow_train", "flow_train_l", "flow_test", "flow_test_l"]
    extensie = ".npy"
    antw = []
    for i in range(0, 4):
        extra = ""
        current = opties[i]
        if augm:
            extra += '_aug'
        if extra_data:
            extra += '_ed'
        if threeD:
            extra += '_3d'
        current += extra + extensie
        if os.path.exists(config.CSV_LOC+current):
            print(f"{current} exits! File will be loaded")
            with open(config.CSV_LOC + current, 'rb') as f:
                t = np.load(f)
                antw.append(t)
        else:
            print(f"{current} does not exist. File will be made")
            flow_train, flow_train_l, flow_test, flow_test_l = getFusionData(config.AantalFrames, augm=augm,
                                                                             extra_fusion=extra_data, threeD=threeD)
            print("Done getting fusion data")
            # flow_train_l = np.array(flow_train_l)
            # flow_test_l = np.array(flow_test_l)
            HF.writeShit('flow_train'+extra+extensie, config.CSV_LOC, flow_train)
            HF.writeShit('flow_train_l'+extra+extensie, config.CSV_LOC, flow_train_l)
            HF.writeShit('flow_test'+extra+extensie, config.CSV_LOC, flow_test)
            HF.writeShit('flow_test_l'+extra+extensie, config.CSV_LOC, flow_test_l)
            return flow_train, flow_train_l, flow_test, flow_test_l
    return antw[0], antw[1], antw[2], antw[3]


def getFusionData(aantal_frames, augm: bool, extra_fusion: bool, threeD = False):
    tv_test_vid, tv_test_label, tv_tr_v, tv_tr_l = train_tests_tv(False)
    tv_tr_l = HF.convertLabel(tv_tr_l)
    tv_test_lab = HF.convertLabel(tv_test_label)

    flow_data, keren = OptF.getVideosFlow2(tv_tr_v, config.TV_VIDEOS_SLASH, True, config.Image_size, aantal_frames,
                                           extra_data=extra_fusion, augm=augm)
    if keren > 1:
        tv_tr_l = HF.double_labels(tv_tr_l, keren)
    flow_data_test = OptF.getVideosFlow(tv_test_vid, config.TV_VIDEOS_SLASH, True, config.Image_size, aantal_frames)
    # tv_train, tv_val, tv_train_l, tv_val_l = train_test_split(flow_data, tv_tr_l, test_size=0.15, stratify=tv_tr_l)

    tv_train_l, tv_test_l = np.array(tv_tr_l), np.array(tv_test_lab)
    return flow_data, tv_train_l, flow_data_test, tv_test_l


#
# Volgens mij worden de functies hieronder niet meer gebruikt
#


def loadStanfordData():
    stf_train_files, stf_train_labels_S, stf_test_files, stf_test_labels = train_test_stanford(False)

    input_shape = (112, 112, 3)
    uniqueLabels, dictionary = HF.getUniques(stf_test_labels)
    stf_train_labels_ind = [dictionary[lab] for lab in stf_train_labels_S]
    stf_test_labels_ind = [dictionary[lab] for lab in stf_test_labels]

    # Run once to get the cropped images
    # HF.convertAndCropImg(stf_train_files, True, True, config.Image_size, config.STANF_CONV_CROP)
    # HF.convertAndCropImg(stf_test_files, True, True, config.Image_size, config.STANF_CONV_CROP)
    # HF.convertNew(stf_train_files, config.Image_size, config.STANF_CONV, config.STANF_CONV_CROP)
    # HF.convertNew(stf_test_files, config.Image_size, config.STANF_CONV, config.STANF_CONV_CROP)
    if config.Use_converted:
        cropped_ = True
        stf_train_imgs = np.array(readConvImages(stf_train_files, cropped=cropped_, grayScale=False))
        stf_test_imgs = np.array(readConvImages(stf_test_files, cropped=cropped_, grayScale=False))

    stf_train_labels = np.array(HF.double_labels(stf_train_labels_ind))
    stf_test_labels = np.array(HF.double_labels(stf_test_labels_ind))

    stf_train_imgs, stf_val_imgs, stf_train_labels, stf_val_labels = train_test_split(stf_train_imgs,
                                                                                      stf_train_labels,
                                                                                      test_size=config.Validate_perc,
                                                                                      stratify=stf_train_labels)

    images_train = np.concatenate([stf_train_imgs, stf_val_imgs])
    labels_train = np.concatenate([stf_train_labels, stf_val_labels])
    test_fold = [-1] * len(stf_train_imgs) + [0] * len(stf_val_imgs)


def readConvImages(imgs, cropped: bool, grayScale: bool):
    lijst = []
    gray = 0 if grayScale else 1
    location = config.STANF_CONV_CROP if cropped else config.STANF_CONV
    for fileName in imgs:
        img = cv2.imread(location + fileName, gray)
        img2 = cv2.imread(location + fileName[:-4] + "_flip.jpg", gray)
        if img is None or img2 is None:
            print(f"None! {fileName}")
        else:
            if grayScale:
                img = img.reshape((config.Image_size, config.Image_size, 1))
                img2 = img2.reshape((config.Image_size, config.Image_size, 1))
            lijst.append(img)
            lijst.append(img2)
    return lijst
