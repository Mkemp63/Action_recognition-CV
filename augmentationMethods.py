import cv2
import tensorflow as tf
from imgaug import augmenters as iaa

# Definitions
blur = iaa.GaussianBlur(sigma=(0.9, 1.0)).to_deterministic()
tx_rechts = iaa.TranslateX(px=(19, 20), mode="reflect").to_deterministic()
tx_links = iaa.TranslateX(px=(-19, -20), mode="reflect").to_deterministic()
high_bright = 0.4
low_bright = -0.4
high_satur = 4
low_satur = 0.5
high_hue = .8
low_hue = .4


def blurImg(img, blur):
    return blur(image=img)


def transposeX(img, tran):
    return tran(image=img)


def grayscaleImg(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def invertImg(img):
    return (255 - img)


def flipImg(img):
    return cv2.flip(img, 1)


def adjustBrightnessImg(img, brightness):
    return tf.image.adjust_brightness(img, brightness).numpy()


def adjustSaturation(img, saturation):
    return tf.image.adjust_saturation(img, saturation).numpy()


def adjustHue(img, hue):
    return tf.image.adjust_hue(img, hue).numpy()
