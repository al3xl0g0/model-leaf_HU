import numpy as np

def dataToCloud(RGB, MS, topleft):
    if (topleft < 2):
        topleft = np.array([1, 1])

    # Convert RGB to double and set 0 to nan
    RGB = RGB.astype(np.double)
    RGB[RGB == 0] = np.nan

    # Convert the multi data to double and set 0 to nan
    MS = MS.astype(np.double)
    MS[MS == 0] = np.nan;

    # RGB-D camera constants
    [height, width, _] = RGB.shape
    center = np.array([height / 2, width / 2])
    matrix = np.true_divide(RGB[:, :, 1], RGB[:, :, 1])

    # Convert depth image to 3d point clouds
    pcloud = np.zeros((height, width, 3))
    xgrid = np.ones((height, 1)) * np.arange(1, width + 1) + (topleft.item(0) - 1) - center.item(0)
    ygrid = np.arange(1, height + 1)[:, None] * np.ones((1, width)) + (topleft.item(1) - 1) - center.item(1)
    pcloud[:, :, 0] = np.true_divide(np.multiply(xgrid, matrix), 100)
    pcloud[:, :, 1] = np.true_divide(np.multiply(ygrid, matrix), 100)
    pcloud[:, :, 2] = MS[:, :]  # np.true_divide(MS[:, :], 1000)

    return pcloud