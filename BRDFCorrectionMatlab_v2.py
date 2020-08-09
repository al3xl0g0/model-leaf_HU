import math
import numpy as np
import cv2
import scipy.signal as signal
from scipy import io
from skimage import io as skio
from DataToCloud import dataToCloud
from DepressionAng import depAng
from DepthToCloud import depthToCloud
from numpy import nan
from pathlib import Path

data_folder = Path(r'C:\Users\Hevra\Downloads\Processed\Ex601_Task_A_1')
depth = skio.imread(data_folder / '3D_Channel_RealSense_Depth.png')
RGB = skio.imread(data_folder / '3D_Channel_RealSense_RGB.jpeg')
MS480_DN = skio.imread(data_folder / 'Multi_Channel_Robin_480.png')
MS520_DN = skio.imread(data_folder / 'Multi_Channel_Robin_520.png')
MS550_DN = skio.imread(data_folder / 'Multi_Channel_Robin_550.png')
MS670_DN = skio.imread(data_folder / 'Multi_Channel_Robin_670.png')
MS700_DN = skio.imread(data_folder / 'Multi_Channel_Robin_700.png')
MS730_DN = skio.imread(data_folder / 'Multi_Channel_Robin_730.png')
MS780_DN = skio.imread(data_folder / 'Multi_Channel_Robin_780.png')


DN2RAD = np.asarray([0.059057, 0.192245, 0.594233, 1.198960, 1.871885, 2.034510, 2.075143])

MS480 = np.multiply(MS480_DN, DN2RAD[0])
MS520 = np.multiply(MS520_DN, DN2RAD[1])
MS550 = np.multiply(MS550_DN, DN2RAD[2])
MS670 = np.multiply(MS670_DN, DN2RAD[3])
MS700 = np.multiply(MS700_DN, DN2RAD[4])
MS730 = np.multiply(MS730_DN, DN2RAD[5])
MS780 = np.multiply(MS780_DN, DN2RAD[6])

# Convert Multi Data image into 3D point cloud
# With dataToCloud_AgriEye function
# Saving the Multi data value

pcloudMS480 = dataToCloud(RGB, MS480, 0)
MS480rad = pcloudMS480[:, :, 2]

pcloudMS520 = dataToCloud(RGB, MS520, 0)
MS520rad = pcloudMS520[:, :, 2]

pcloudMS550 = dataToCloud(RGB, MS550, 0)
MS550rad = pcloudMS550[:, :, 2]

pcloudMS670 = dataToCloud(RGB, MS670, 0)
MS670rad = pcloudMS670[:, :, 2]

pcloudMS700 = dataToCloud(RGB, MS700, 0)
MS700rad = pcloudMS700[:, :, 2]

pcloudMS730 = dataToCloud(RGB, MS730, 0)
MS730rad = pcloudMS730[:, :, 2]

pcloudMS780 = dataToCloud(RGB, MS780, 0)
MS780rad = pcloudMS780[:, :, 2]

# BRDF Correction
# % extract row and column size from z
# filter2 = signal.medfilt(z)
# % smoth the outliners in z by using medfilter function of matlab
# % were each output pixel contains the median value in the 3 - by - 3
# % neighborhood  arround the croosponding pixel in the input image (z)
# % Filter2 = signal.medfilt(z)
# % Change all values that are les than 1
# % from the Filted image
# % the sensor is 1m above the object
[pcloud, distance] = depthToCloud(RGB, depth, 0)
z = pcloud[:, :, 2]
[r, c] = np.shape(z)
# MASK
Filter2 = signal.medfilt(z)
mask = MS480rad

for i in range(r):
    for j in range(c):
        if mask[i, j] > 0:
            mask[i, j] = 1
        else:
            mask[i, j] = 0

mask = mask.astype(np.int)
Filter2 = np.multiply(Filter2, mask)
dang = np.zeros(np.shape(z))

for i in range(r):
    for j in range(c):
        if Filter2[i, j] >= 0.8:
            Filter2[i, j] = Filter2[i, j]
        else:
            Filter2[i, j] = nan

# Filter2 array upside down flip flup it
Filter2 = np.flipud(Filter2)

# The minimal distance between the sansor to object
H_min = np.nanmin(Filter2)
# Calculate the depth Coefficient
H = np.true_divide(Filter2, H_min)
# Implaement the depth coefficient om MultiSpectral Data from
Rad2d_depth480 = np.multiply(MS480rad, H)
Rad2d_depth520 = np.multiply(MS520rad, H)
Rad2d_depth550 = np.multiply(MS550rad, H)
Rad2d_depth670 = np.multiply(MS670rad, H)
Rad2d_depth700 = np.multiply(MS700rad, H)
Rad2d_depth730 = np.multiply(MS730rad, H)
Rad2d_depth780 = np.multiply(MS780rad, H)

for i in range(r):
    for j in range(c):
        if math.isnan(Filter2[i, j]):
            dang[i, j] = 0
        else:
            dang[i, j] = depAng(1, Filter2[i, j])

# ALEX, Need to ADD: depAng(imag(depAng) ~= 0) = 0;

for i in range(r):
    for j in range(c):
        if np.imag(dang[i, j]) != 0:
            dang[i, j] = 0
        else:
            dang[i, j] = dang[i, j]#depAng(1, dang[i, j])


#dang = np.imag(dang)
# Generating new matrix
# in size of the origenal image (z)
# and the starting value is 90 to all cells
coef = np.ones((r, c))
coef = coef * 90
angcoef = np.ones((r, c))

for i in range(r):
    for j in range(c):
        if dang[i, j] < 45:
            angcoef[i, j] = nan
        else:
            angcoef[i, j] = np.true_divide(coef[i, j], dang[i, j])

#angcoef = np.flipud(angcoef)
# Calculat parameters to 3D correction in 2 steps%%%
# Coefficient based on polynom calculation from SPHER
# Alignment each of the pixels  - the small Align
# Test
Rad3dang_480 = np.multiply(-763, np.power(angcoef, 2)) + np.multiply(1956.8, angcoef) - 1238.9
Rad3dang_520 = np.multiply(-972.69, np.power(angcoef, 2)) + np.multiply(2505.9, angcoef) - 1603.2
Rad3dang_550 = np.multiply(-383.73, np.power(angcoef, 2)) + np.multiply(954.5, angcoef) - 578.42
Rad3dang_670 = np.multiply(-309.5, np.power(angcoef, 2)) + np.multiply(745.36, angcoef) - 432.14
Rad3dang_700 = np.multiply(-177.96, np.power(angcoef, 2)) + np.multiply(422.32, angcoef) - 237.6
Rad3dang_730 = np.multiply(-250.67, np.power(angcoef, 2)) + np.multiply(633.22, angcoef) - 391.33
Rad3dang_780 = np.multiply(-360.1, np.power(angcoef, 2)) + np.multiply(897.57, angcoef) - 548.05

# Seconde Align - to the sensor

# Rad3dang_coef480 = (max(max(Rad3dang_480)) + (max(max(Rad3dang_480)) - Rad3dang_480)). / Rad3dang_480;


Rad3dang_coef480 = np.true_divide(np.add(np.nanmax(Rad3dang_480), np.subtract(np.nanmax(Rad3dang_480), Rad3dang_480)),
                                  Rad3dang_480)
Rad3dang_coef520 = np.true_divide(np.add(np.nanmax(Rad3dang_520), np.subtract(np.nanmax(Rad3dang_520), Rad3dang_520)),
                                  Rad3dang_520)
Rad3dang_coef550 = np.true_divide(np.add(np.nanmax(Rad3dang_550), np.subtract(np.nanmax(Rad3dang_550), Rad3dang_550)),
                                  Rad3dang_550)
Rad3dang_coef670 = np.true_divide(np.add(np.nanmax(Rad3dang_670), np.subtract(np.nanmax(Rad3dang_670), Rad3dang_670)),
                                  Rad3dang_670)
Rad3dang_coef700 = np.true_divide(np.add(np.nanmax(Rad3dang_700), np.subtract(np.nanmax(Rad3dang_700), Rad3dang_700)),
                                   Rad3dang_700)
Rad3dang_coef730 = np.true_divide(np.add(np.nanmax(Rad3dang_730), np.subtract(np.nanmax(Rad3dang_730), Rad3dang_730)),
                                   Rad3dang_730)
Rad3dang_coef780 = np.true_divide(np.add(np.nanmax(Rad3dang_780), np.subtract(np.nanmax(Rad3dang_780), Rad3dang_780)),
                                   Rad3dang_780)





# % 3D correction based on BRDF
# % project each pixel to one plane
Rad3dang_corr480 = np.ones(np.shape(z))
Rad3dang_corr520 = np.ones(np.shape(z))
Rad3dang_corr550 = np.ones(np.shape(z))
Rad3dang_corr670 = np.ones(np.shape(z))
Rad3dang_corr700 = np.ones(np.shape(z))
Rad3dang_corr730 = np.ones(np.shape(z))
Rad3dang_corr780 = np.ones(np.shape(z))



for i in range(r):
    for j in range(c):
        if Rad3dang_730[i, j] > 1:
            Rad3dang_corr730[i, j] = Rad2d_depth730[i, j]
        else:
            Rad3dang_corr730[i, j] = np.multiply(Rad2d_depth730[i, j], Rad3dang_coef730[i, j])

        # if Rad3dang_520[i, j] <= 1 or Rad3dang_coef520[i, j] > 2:
        #     Rad3dang_corr520[i, j] = Rad2d_depth520[i, j]
        # else:
        #     Rad3dang_corr520[i, j] = np.multiply(Rad2d_depth520[i, j], Rad3dang_coef520[i, j])
        #
        # if Rad3dang_550[i, j] <= 1 or Rad3dang_coef550[i, j] > 2:
        #     Rad3dang_corr550[i, j] = Rad2d_depth550[i, j]
        # else:
        #     Rad3dang_corr550[i, j] = np.multiply(Rad2d_depth550[i, j], Rad3dang_coef550[i, j])
        #
        # if Rad3dang_670[i, j] <= 1 or Rad3dang_coef670[i, j] > 2:
        #     Rad3dang_corr670[i, j] = Rad2d_depth670[i, j]
        # else:
        #     Rad3dang_corr670[i, j] = np.multiply(Rad2d_depth670[i, j], Rad3dang_coef670[i, j])
        #
        # if Rad3dang_700[i, j] <= 1 or Rad3dang_coef700[i, j] > 2:
        #     Rad3dang_corr700[i, j] = Rad2d_depth700[i, j]
        # else:
        #     Rad3dang_corr700[i, j] = np.multiply(Rad2d_depth700[i, j], Rad3dang_coef700[i, j])
        #
        # if Rad3dang_730[i, j] <= 1 or Rad3dang_coef730[i, j] > 2:
        #     Rad3dang_corr730[i, j] = Rad2d_depth730[i, j]
        # else:
        #     Rad3dang_corr730[i, j] = np.multiply(Rad2d_depth730[i, j], Rad3dang_coef730[i, j])
        #
        # if Rad3dang_780[i, j] <= 1 or Rad3dang_coef780[i, j] > 2:
        #     Rad3dang_corr780[i, j] = Rad2d_depth780[i, j]
        # else:
        #     Rad3dang_corr780[i, j] = np.multiply(Rad2d_depth480[i, j], Rad3dang_coef780[i, j])



# cv2.imwrite(r'C:\Users\Hevra\Downloads\Processed\Ex601_Task_A_1\V2_Rad3dang_480_cv2.png', Rad3dang_corr480.astype(np.uint16), [cv2.IMWRITE_PNG_COMPRESSION, 0])
# cv2.imwrite(r'C:\Users\Hevra\Downloads\Processed\Ex601_Task_A_1\V2_Rad3dang_550_cv2.png', Rad3dang_corr550.astype(np.uint16), [cv2.IMWRITE_PNG_COMPRESSION, 0])
# cv2.imwrite(r'C:\Users\Hevra\Downloads\Processed\Ex601_Task_A_1\V2_Rad3dang_670_cv2.png', Rad3dang_corr670.astype(np.uint16), [cv2.IMWRITE_PNG_COMPRESSION, 0])
# cv2.imwrite(r'C:\Users\Hevra\Downloads\Processed\Ex601_Task_A_1\V2_Rad3dang_700_cv2.png', Rad3dang_corr700.astype(np.uint16), [cv2.IMWRITE_PNG_COMPRESSION, 0])
cv2.imwrite(r'C:\Users\Hevra\Downloads\Processed\Ex601_Task_A_1\V2_Rad3dang_730_cv2.png', Rad3dang_corr730.astype(np.uint16), [cv2.IMWRITE_PNG_COMPRESSION, 0])
# cv2.imwrite(r'C:\Users\Hevra\Downloads\Processed\Ex601_Task_A_1\V2_Rad3dang_780_cv2.png', Rad3dang_corr780.astype(np.uint16), [cv2.IMWRITE_PNG_COMPRESSION, 0])
#
