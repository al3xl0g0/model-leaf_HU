import signal
import numpy as np
import math

from DepressionAng import depAng
from DepthToCloud import depthToCloud

[pcloud, distance] = depthToCloud(RGB, depth, 0)
z = pcloud[:,:,2]
[r,c] = np.shape(z)
Filter2 = signal.medfilt(z)
dang = np.zeros(np.shape(z))

for i in range(r):
  for j in range(c):
    if Filter2[i,j] >= 1:
        Filter2[i,j] = Filter2[i,j]
    else:
      Filter2[i,j] = nan

for i in range(r):
  for j in range(c):
    if math.isnan(Filter2[i, j]):
      dang[i, j] = 0
    else:
      dang[i, j] = depAng(1, Filter2[i, j])

print(dang)