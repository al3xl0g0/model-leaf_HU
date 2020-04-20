import numpy as np

def depthToCloud(RGB, depth, topleft):
  if (topleft < 2):
    topleft = np.array([1,1])

  # Convert RGB to double and set 0 to nan
  RGB = RGB.astype(np.double)
  RGB[RGB==0] = np.nan

  # Convert depth to double and set 0 to nan
  depth = depth.astype(np.double)
  depth[depth==0] = np.nan

  # RGB-D camera constants
  [height, width, _] = RGB.shape
  center = np.array([height / 2, width / 2])
  matrix = np.true_divide(depth, depth)

  # Convert depth image to 3d point clouds
  pcloud = np.zeros((height, width, 3))
  xgrid = np.arange(1, height + 1)[:, None] * np.ones((1, width)) + (topleft.item(0) - 1) - center.item(0)
  ygrid = np.ones((height, 1)) * np.arange(1, width + 1) + (topleft.item(1) - 1) - center.item(1)
  pcloud[:, :, 0] = np.true_divide(np.multiply(xgrid, matrix), 100)
  pcloud[:, :, 1] = np.true_divide(np.multiply(ygrid, matrix), 100)
  pcloud[:, :, 2] = np.flipud(np.true_divide(depth[:, :], 1000))
  distance = np.sqrt(np.sum(np.power(pcloud, 2), 2))

  return [pcloud, distance]