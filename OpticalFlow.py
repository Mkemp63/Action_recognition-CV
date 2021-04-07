import cv2
import numpy as np
import os
import math
import config
import HelperFunctions as HF


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
            # test = cropImg(frame, video_size[1])
            # cv2.imshow("crop", test)

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


def getOpticalFlowVideo(location: str, file: str, numbOfFrames: int, crop: bool, size: int = -1):
    ans = [] # numbOfFrames aantal frames waarbij ans[f*2] = magnitude, ans[f*2+1] = angle

    cap = cv2.VideoCapture(location + file)
    video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # every_I = int(math.floor((length-1)/numbOfFrames))
    every_I = (length-3)/numbOfFrames

    ret, first_frame = cap.read()
    frame = transformFrame(first_frame, video_size[1], crop, size)
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    i = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = transformFrame(frame, video_size[1], crop, size)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if i == round((len(ans)/2+1)*every_I) and len(ans) < 20:  # if i % every_I == 0:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            ans.append(magnitude)
            ans.append(angle)

        prev_gray = gray
        i += 1

    cap.release()
    if len(ans) != 20:
        print(f"numb of frames: {length} vs {i} & every: {every_I}; {(length-1)/numbOfFrames}")
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
        ans.append(np.transpose(np.array(a), (1,2,0)))
    return np.array(ans)

# test()
# calcOpticalFlow("J:\\Python computer vision\\Action_recognition-CV\\Data\\TV-HI\\tv_human_interactions_videos\\",
#                 "kiss_0045.avi", True, True)
# convertVideos("tv_human_interactions_videos", config.Image_size)

# ans = getOpticalFlowVideo("J:\\Python computer vision\\Action_recognition-CV\\Data\\TV-HI\\tv_human_interactions_videos\\",
#                           "kiss_0045.avi", 10, True, 112)
# print(type(ans))
# print(type(ans[0]))
# print(len(ans))
# print(ans[0].shape)
# twee = np.transpose(np.array(ans), (1,2,0))
# print(type(twee))
# print(twee.shape)