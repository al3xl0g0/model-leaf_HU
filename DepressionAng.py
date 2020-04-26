from math import *
import numpy as np

#depAng = sin-1((H^2 + 2HRe + R^2)/(2R(H + Re)))
def asind(x):
    rad = np.arcsin(x)
    return np.degrees(rad)

def depAng(H, R, Re=8.4774e+06): #changed from 4/3 radius of Earth
    H2 = np.multiply(2, H)
    H2Re = np.multiply(H2, Re)
    R2 = np.multiply(2, R)

    return asind(np.true_divide((H ** 2 + H2Re + R ** 2), (R2 * (H + Re))))

