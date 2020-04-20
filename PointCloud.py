import numpy as np
import open3d as o3d


def reshape(matrix):
  x = matrix[:,:,0]
  y = matrix[:,:,1]
  z = matrix[:,:,2] # np.multiply(matrix[:,:,2], 1000)
  xyz = np.zeros((np.size(x), 3))
  xyz[:, 0] = x.reshape(-1, order='F').copy()
  xyz[:, 1] = y.reshape(-1, order='F').copy()
  xyz[:, 2] = z.reshape(-1, order='F').copy()

  return xyz

def toPointCloud(pcloud, RGB):
  xyz = reshape(pcloud)
  xyzRGB = reshape(RGB)
  ptCloud = o3d.geometry.PointCloud()
  ptCloud.points = o3d.utility.Vector3dVector(xyz)
  ptCloud.colors = o3d.utility.Vector3dVector(xyzRGB)

  return ptCloud