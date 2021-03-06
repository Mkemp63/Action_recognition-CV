import cv2
import numpy as np
import os
import math
import config
import HelperFunctions as HF
import tensorflow as tf
import augmentationMethods as am


def calcOpticalFlow(location: str, file: str, write: bool, view: bool = False,
                    saveLoc: str = os.path.join(config.TV, "flowVids\\")):
    cap = cv2.VideoCapture(location + file)

    ret, first_frame = cap.read()
    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less expensive
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(first_frame)
    mask[..., 1] = 255

    if write:
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        out = cv2.VideoWriter(saveLoc + file, fourcc, fps, video_size)
        print(f"Size: {mask.shape}")
        print(f"FOURCC: {int(cap.get(cv2.CAP_PROP_FOURCC))}")
        print(f"FPS: {fps} & FOURCC: {fourcc}")
        print(f"LENGTH: {length}")
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if view:
            cv2.imshow("input", frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        print(f"Mag: {magnitude[208][288]} & angle: {angle[208][288]}")
        mask[..., 0] = angle * 180 / np.pi / 2

        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
        if view:
            cv2.imshow("dense optical flow", rgb)
            print(f"mag: {magnitude.shape} & angle: {angle.shape}")
        prev_gray = gray

        if ret and write:
            # print(i)
            i += 1
            out.write(rgb)
        if view:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    # close windows and stuff
    cap.release()
    if write:
        out.release()
    cv2.destroyAllWindows()


def test():
    img = cv2.imread(os.path.join(config.STANF_IMG, 'applauding_005.jpg'), 0)
    if img is None:
        print("NONE")
    cv2.startWindowThread()
    cv2.namedWindow("img")
    cv2.imshow("img", img)
    cv2.waitKey(0)
    test = HF.cropImg(img, 300)
    cv2.imshow("crop", test)
    cv2.waitKey(0)
    input()
    cv2.destroyAllWindows()


def transformFrame(frame, h: int, crop: bool, size: int):
    if crop and size != -1:
        return cv2.resize(HF.cropImg(frame, h), (size, size))
    elif crop:
        return HF.cropImg(frame, h)
    elif size != -1:
        return cv2.resize(frame, (size, size))
    return frame


def augmentFrame(img, augm_num):
    if augm_num == 0:
        return img
    elif augm_num == 1:
        return am.blurImg(img, am.blur)
    elif augm_num == 2:
        return am.transposeX(img, am.tx_rechts)
    elif augm_num == 3:
        return am.transposeX(img, am.tx_links)
    elif augm_num == 4:
        return am.grayscaleImg(img)
    elif augm_num == 5:
        return am.invertImg(img)
    elif augm_num == 6:
        return am.flipImg(img)
    elif augm_num == 7:
        return am.adjustBrightnessImg(img, am.high_bright)
    elif augm_num == 8:
        return am.adjustBrightnessImg(img, am.low_bright)
    elif augm_num == 9:
        return am.adjustSaturation(img, am.high_satur)
    elif augm_num == 10:
        return am.adjustSaturation(img, am.low_satur)
    elif augm_num == 11:
        return am.adjustSaturation(img, am.low_hue)


def getOpticalFlowVideo(location: str, file: str, numbOfFrames: int, crop: bool, size: int = -1, augm_num: int = 0,
                        addEffect: bool = False, extra_i: int = 0):
    ans = []  # numbOfFrames aantal frames waarbij ans[f*2] = magnitude, ans[f*2+1] = angle

    cap = cv2.VideoCapture(location + file)
    video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    every_I = (length-3)/numbOfFrames

    ret, first_frame = cap.read()
    frame = transformFrame(first_frame, video_size[1], crop, size)
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    i = 1

    extra = extra_i * int(((length-3)/numbOfFrames / 2))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = transformFrame(frame, video_size[1], crop, size)
        frame = augmentFrame(frame, augm_num)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if i == round((len(ans)/2+1)*every_I) - extra and len(ans) < 2*numbOfFrames:  # if i % every_I == 0:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            ans.append(magnitude)
            ans.append(angle)

        prev_gray = gray
        i += 1

    cap.release()
    if len(ans) != 2*numbOfFrames:
        print(f"numb of frames: {length} vs {i} & every: {every_I}; {(length-1)/numbOfFrames};    {extra_i}, {extra}")
    return ans


def convertVideo(location: str, file: str, saveLoc: str, size: int, resize: bool = True, crop: bool = False):
    cap = cv2.VideoCapture(location + file)

    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    out = cv2.VideoWriter(saveLoc + file, fourcc, fps, (size, size))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if resize and crop:
            img1 = HF.cropImg(frame, h)
            img = cv2.resize(img1, (size, size))
        elif resize:
            img = cv2.resize(frame, (size, size))
        else:
            img = HF.cropImg(frame, size)
        out.write(img)

        cv2.imshow("input", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # close windows and stuff
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def convertVideos(loc: str, size: int, saveLoc: str = "rightSize\\",
                  baseLoc: str = config.TV):
    location = baseLoc + loc
    i = 0
    for fileName in os.listdir(location):
        if fileName[0:3] != "neg":
            convertVideo(location + "\\", fileName, baseLoc + saveLoc, size)
        print(f"{i}: {fileName}")
    print("DONE!")


def getVideosFlow(files, location: str, crop: bool, size: int,  count: int = 10):
    ans = []
    for fileName in files:
        a = getOpticalFlowVideo(location, fileName, count, crop, size)
        if len(a) != 20:
            print(f"Length: {len(a)}")
        ans.append(np.transpose(np.array(a), (1, 2, 0)))
    return np.array(ans)


def getVideosFlow2(files, location: str, crop: bool, size: int,  count: int = 10,
                   extra_data: bool = False, augm: bool = False):
    ans = []
    augmIndexes = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11]  # 4 is grayscale
    rondes = 2 if extra_data else 1
    for fileName in files:
        print(fileName)
        for i in range(0, rondes):
            a = getOpticalFlowVideo(location, fileName, count, crop, size, extra_i=i)
            if len(a) != 2*count:
                print(f"Length: {len(a)}")
            ans.append(np.transpose(np.array(a), (1, 2, 0)))
            if augm:
                for j in range(0, len(augmIndexes)):
                    a = getOpticalFlowVideo(location, fileName, count, crop, size, extra_i=i, augm_num=augmIndexes[j])
                    if len(a) != 2*count:
                        print(f"Length: {len(a)}")
                    ans.append(np.transpose(np.array(a), (1, 2, 0)))
    aantal = rondes*(len(augmIndexes)+1) if augm else rondes
    return np.array(ans), aantal
