from typing import Tuple
import cv2
import numpy as np
import os


def show_image_cv2(image, name: str = "image"):
    cv2.imshow(name, image)


def resize_image_cv2(img, size: Tuple[int, int]):
    return cv2.resize(img, size)


def read_image_cv2(path: str):
    return cv2.imread(path)


def save_image_cv2(img, path: str):
    dir = "/".join(path.split("/")[:-1])
    os.makedirs(dir, exist_ok=True)
    return cv2.imwrite(path, img)


def preprocess_image_cv2(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_val = np.array([0, 0, 0])
    upper_val = np.array([179, 255, 127])
    mask = cv2.inRange(hsv, lower_val, upper_val)
    # invert bits to get black letter on white background
    mask_inv = cv2.bitwise_not(mask)
    return mask_inv
