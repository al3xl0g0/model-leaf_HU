import math
import numpy as np
import scipy.signal as signal
from numpy import nan
from skimage import io as skio

from DataToCloud import dataToCloud
from DepressionAng import depAng
from DepthToCloud import depthToCloud

depth = skio.imread('/content/gdrive/My Drive/Processed/3D_Channel_RealSense_Depth.png')
RGB = skio.imread('/content/gdrive/My Drive/Processed/3D_Channel_RealSense_RGB.jpeg')

MS480_DN = skio.imread('/content/gdrive/My Drive/Processed/Multi_Channel_Robin_480.png');
MS520_DN = skio.imread('/content/gdrive/My Drive/Processed/Multi_Channel_Robin_520.png');
MS550_DN = skio.imread('/content/gdrive/My Drive/Processed/Multi_Channel_Robin_550.png');
MS670_DN = skio.imread('/content/gdrive/My Drive/Processed/Multi_Channel_Robin_670.png');
MS700_DN = skio.imread('/content/gdrive/My Drive/Processed/Multi_Channel_Robin_700.png');
MS730_DN = skio.imread('/content/gdrive/My Drive/Processed/Multi_Channel_Robin_730.png');
MS780_DN = skio.imread('/content/gdrive/My Drive/Processed/Multi_Channel_Robin_780.png');
thermal = skio.imread('/content/gdrive/My Drive/Processed/Thermal_Channel_0.png');

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
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%% BRDF Correction %%%%%%%
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
Filter2 = signal.medfilt(z)
dang = np.zeros(np.shape(z))

for i in range(r):
    for j in range(c):
        if Filter2[i, j] >= 1:
            Filter2[i, j] = Filter2[i, j]
        else:
            Filter2[i, j] = nan

# Filter2 array upside down flip flup it
Filter2 = np.flipud(Filter2)

# The minimal distance between the sansor to object
# H_min = np.double(np.min(np.min(Filter2)))
## TO-DO extract min vlue from Filter2
H_min = 1
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

# Generating new metrix
# in size of the origenal image (z)
# and the starting value is 90 to all cells
coef = np.ones((r, c))
coef = coef * 90
angcoef = np.ones((r, c))

for i in range(r):
    for j in range(c):
        if dang[i, j] < 60:
            angcoef[i, j] = nan
        else:
            angcoef[i, j] = np.true_divide(coef[i, j], dang[i, j])
# Calculat parameters to 3D correction in 2 steps%%%
# Coefficient based on polynom calculation from SPHER
# Alignment each of the pixels  - the small Align

Rad3dang_480 = np.multiply(-3.0929, np.power(angcoef, 2)) + np.multiply(4.4708, angcoef) - 0.4843
Rad3dang_520 = np.multiply(1.9053, np.power(angcoef, 2)) - np.multiply(6.8177, angcoef) + 5.8257
Rad3dang_550 = np.multiply(-1.4214, np.power(angcoef, 2)) + np.multiply(1.2813, angcoef) + 0.8046
Rad3dang_670 = np.multiply(1.1746, np.power(angcoef, 2)) - np.multiply(3.6691, angcoef) + 2.8396
Rad3dang_700 = np.multiply(1.4149, np.power(angcoef, 2)) - np.multiply(3.9761, angcoef) + 2.8536
Rad3dang_730 = np.multiply(0.4538, np.power(angcoef, 2)) - np.multiply(1.6242, angcoef) + 1.4194
Rad3dang_780 = np.subtract(np.multiply(-0.1729, np.power(angcoef, 2)), np.add(np.multiply(0.3105, angcoef), 0.7748))

# Seconde Align - to the sensor
Rad3dang_coef480 = 1. + 1. - Rad3dang_480
Rad3dang_coef520 = 1. + 1. - Rad3dang_520
Rad3dang_coef550 = 1. + 1. - Rad3dang_550
Rad3dang_coef670 = 1. + 1. - Rad3dang_670
Rad3dang_coef700 = 1. + 1. - Rad3dang_700
Rad3dang_coef730 = 1. + 1. - Rad3dang_730
Rad3dang_coef780 = 1. + 1. - Rad3dang_780

# % 3D correction based on BRDF
# % project each pixel to one plane

Rad3d_corr480 = np.multiply(Rad2d_depth480, Rad3dang_coef480)
Rad3d_corr520 = np.multiply(Rad2d_depth520, Rad3dang_coef520)
Rad3d_corr550 = np.multiply(Rad2d_depth550, Rad3dang_coef550)
Rad3d_corr670 = np.multiply(Rad2d_depth670, Rad3dang_coef670)
Rad3d_corr700 = np.multiply(Rad2d_depth700, Rad3dang_coef700)
Rad3d_corr730 = np.multiply(Rad2d_depth730, Rad3dang_coef730)
Rad3d_corr780 = np.multiply(Rad2d_depth780, Rad3dang_coef780)

# Second Align - to the sensor


# 3D correction based on BRDF project each pixel to one plane

Rad3d_corr520 = np.multiply(Rad2d_depth520, Rad3dang_coef520)
Rad3d_corr550 = np.multiply(Rad2d_depth550, Rad3dang_coef550)
Rad3d_corr670 = np.multiply(Rad2d_depth670, Rad3dang_coef670)
Rad3d_corr700 = np.multiply(Rad2d_depth700, Rad3dang_coef700)
Rad3d_corr730 = np.multiply(Rad2d_depth730, Rad3dang_coef730)
Rad3d_corr780 = np.multiply(Rad2d_depth780, Rad3dang_coef780)

# Radiance2Reflectance
# Gain coefficients calculated from lab experiment for each Band
# In order to bring the corent RADIANCE data to Reflectance
# Coefficients to Banana in Rahan
Gain = np.asarray([0.0428, 0.0301, 0.0179, 0.0056, 0.0089, 0.0083, 0.0074])

Ref3d_corr480 = np.multiply(Rad3d_corr480, Gain[0])
Ref3d_corr520 = np.multiply(Rad3d_corr520, Gain[1])
Ref3d_corr550 = np.multiply(Rad3d_corr550, Gain[2])
Ref3d_corr670 = np.multiply(Rad3d_corr670, Gain[3])
Ref3d_corr700 = np.multiply(Rad3d_corr700, Gain[4])
Ref3d_corr730 = np.multiply(Rad3d_corr730, Gain[5])
Ref3d_corr780 = np.multiply(Rad3d_corr780, Gain[6])

#Test Matlab to python
print(Ref3d_corr780[421, 422])
