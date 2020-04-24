from math import *
import numpy as np


# % depAng = sin-1((H^2 + 2HRe + R^2)/(2R(H + Re)))

def depAng(H, R, Re=4 / 3):
    H2 = np.multiply(2, H)
    H2Re = np.multiply(H2, Re)
    R2 = np.multiply(2, R)

    return asin(np.true_divide((H ** 2 + H2Re + R ** 2), (R2 * (H + Re))))


print(depAng(1, 1))
