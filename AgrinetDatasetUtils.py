
import math

import cv2
import numpy as np
from PIL import Image
from skimage import exposure

from DataToCloud import dataToCloud


def calc_projection_image (ptCloud, pcloud, messyValidRGB):
    ptCloudPoints = np.asarray(ptCloud.points)
    validIndices = np.argwhere(
        np.isfinite(ptCloudPoints[:, 0]) & np.isfinite(ptCloudPoints[:, 1]) & np.isfinite(ptCloudPoints[:, 2]))
    count = np.asarray(ptCloud.points).shape[0]
    indices = np.arange(0, count)
    [u, v] = np.unravel_index(indices, (pcloud.shape[0], pcloud.shape[1]), order='F')
    imagePoints = np.column_stack((u[validIndices], v[validIndices]))

    projImage = np.zeros((np.max(imagePoints[:, 0]) + 1, np.max(imagePoints[:, 1]) + 1, 3))

    for i in range(imagePoints.shape[0]):
        projImage[imagePoints[i, 0], imagePoints[i, 1], :] = messyValidRGB[i, :]
    projImage = projImage.astype(np.uint8)

    return projImage




def dn_to_rad(RGB, MS480, MS520, MS550, MS670, MS700, MS730, MS780):
    # Ratio calculated from old Banana MS image and new banana MS image for each Band
    # In order to bring the corent RAW data to RADIANCE
    # Coefficients to Banana in Rahan

    DN2RAD = [0.059057, 0.192245, 0.594233, 1.198960, 1.871885, 2.034510, 2.075143];

    MS480 = np.multiply(MS480, DN2RAD[0])
    MS520 = np.multiply(MS520, DN2RAD[1])
    MS550 = np.multiply(MS550, DN2RAD[2])
    MS670 = np.multiply(MS670, DN2RAD[3])
    MS700 = np.multiply(MS700, DN2RAD[4])
    MS730 = np.multiply(MS730, DN2RAD[5])
    MS780 = np.multiply(MS780, DN2RAD[6])

    # Convert Multi Data image into 3D point cloud
    # With dataToCloud_AgriEye function
    # Saving the Multi data value
    pcloudMS480 = dataToCloud(RGB, MS480, 0);
    MS480rad = pcloudMS480[:, :, 2];

    pcloudMS520 = dataToCloud(RGB, MS520, 0);
    MS520rad = pcloudMS520[:, :, 2];

    pcloudMS550 = dataToCloud(RGB, MS550, 0);
    MS550rad = pcloudMS550[:, :, 2];

    pcloudMS670 = dataToCloud(RGB, MS670, 0);
    MS670rad = pcloudMS670[:, :, 2];

    pcloudMS700 = dataToCloud(RGB, MS700, 0);
    MS700rad = pcloudMS700[:, :, 2];

    pcloudMS730 = dataToCloud(RGB, MS730, 0);
    MS730rad = pcloudMS730[:, :, 2];

    pcloudMS780 = dataToCloud(RGB, MS780, 0);
    MS780rad = pcloudMS780[:, :, 2];

    MSradlist = [MS480rad, MS520rad, MS550rad, MS670rad, MS700rad, MS730rad, MS780rad]

    return MSradlist

def enhance(MS670rad, MS550rad, MS480rad):
    cat = np.concatenate((MS670rad, MS550rad, MS480rad), axis=1) #AXIS CHANGED FROM 2 TO 1
    img = cat.astype(np.uint8)
    # Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))

    return exposure.rescale_intensity(img, in_range=(p2, p98))

def bdrf_correction():
    pass

def calc_3d_correction():

    pass

def radians_to_reflectance(Rad3d_corr480, Rad3d_corr520, Rad3d_corr550, Rad3d_corr670, Rad3d_corr700, Rad3d_corr730, Rad3d_corr780 ):
    Gain = [0.0428, 0.0301, 0.0179, 0.0056, 0.0089, 0.0083, 0.0074];

    Ref3d_corr480 = Rad3d_corr480 * Gain(0);
    Ref3d_corr520 = Rad3d_corr520 * Gain(1);
    Ref3d_corr550 = Rad3d_corr550 * Gain(2);
    Ref3d_corr670 = Rad3d_corr670 * Gain(3);
    Ref3d_corr700 = Rad3d_corr700 * Gain(4);
    Ref3d_corr730 = Rad3d_corr730 * Gain(5);
    Ref3d_corr780 = Rad3d_corr780 * Gain(6);

    pass

def ver_3d_corrected_brdf():

    #Verification of 3D corrected based BRDF
    pass






def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH))


def add_image(img1, img2, x_center, y_center, x_scale, y_scale, angle):
    img2 = img2.resize((int(x_scale * img2.size[0]), int(y_scale * img2.size[1])), resample=Image.BICUBIC) # Image.ANTIALIAS
    img2 = img2.rotate(angle, resample=Image.BICUBIC, expand=True)

    rows, cols, channels = np.asarray(img2).shape
    x_from = x_center - math.floor(cols / 2.)
    y_from = y_center - math.floor(rows / 2.)

    img1.paste(img2, (x_from, y_from), img2)
    # tmp_mask = image_to_mask(img2)
    # tmp_mask = Image.fromarray(tmp_mask)
    # img1.paste(img2, (x_from, y_from), tmp_mask)

    return img1

def image_to_mask(image):
    # AZ TODO: check if OK when image is 2D (grayscale)
    img_sum = np.sum(image, axis=-1)
    mask = img_sum > 0
    #se = scipy.ndimage.generate_binary_structure(2, 1)
    #mask = scipy.ndimage.binary_erosion(mask, structure=se, iterations = 2)
    return mask

def mask_to_image(mask):
    x, y, z = mask.shape
    image = np.zeros((x, y), dtype=np.uint8)
    for i in range(0, z):
        mask_color = int(((i + 1) / z) * 255)
        image += mask[:, :, i] * np.cast[np.uint8](mask_color)
    return image

def add_image_without_transparency(img1, img2, x_center, y_center, x_scale, y_scale, angle):
    img2 = cv2.resize(img2, None, fx=x_scale, fy=y_scale, interpolation=cv2.INTER_CUBIC)

    img2 = rotate_bound(img2, 360 - angle)

    rows, cols, channels = img2.shape
    x_from = x_center - math.floor(cols / 2.)
    x_to = x_center + math.ceil(cols / 2.)
    y_from = y_center - math.floor(rows / 2.)
    y_to = y_center + math.ceil(rows / 2.)

    y_max, x_max, _ = img1.shape

    if x_from < 0:
        img2 = img2[:, -x_from:]
        x_from = 0
    if x_to >= x_max:
        img2 = img2[:, :-(x_to - x_max + 1)]
        x_to = x_max - 1
    if y_from < 0:
        img2 = img2[-y_from:, :]
        y_from = 0
    if y_to >= y_max:
        img2 = img2[:-(y_to - y_max + 1), :]
        y_to = y_max - 1

    roi = img1[y_from:y_to, x_from:x_to]

    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
    #dst = cv2.add(img1_bg, img2_fg[:, :, :])  # AZ (remove alpha)
    dst = cv2.add(img1_bg, img2_fg[:, :, 0:3])
    img1[y_from:y_to, x_from:x_to] = dst
    return img1
