import os
import pywt
import numpy as np
import matplotlib.pyplot as plt
from math import log10, sqrt


def rgb2gray(rgb):
    if len(rgb.shape) > 2:
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return rgb

def PSNR(image1, image2): 
    if image1.size < image2.size:
        minimal_width = image1.shape[0]
        minimal_length = image1.shape[1]
    else:
        minimal_width = image2.shape[0]
        minimal_length = image2.shape[1]
        
    image1 = np.resize(image1, (minimal_width, minimal_length))
    image2 = np.resize(image2, (minimal_width, minimal_length))
    mse = np.mean((image1 - image2) ** 2) 
    if(mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr

def show(row, column, img_num, image, title):
    plt.subplot(row, column, img_num)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title(title)
    
def wt_coeff(image1, image2, wavelet, max_or_mean):
    coeff1 = pywt.dwt2(image1, wavelet)
    coeff2 = pywt.dwt2(image2, wavelet)

    cA1, (cH1, cV1, cD1) = coeff1
    cA2, (cH2, cV2, cD2) = coeff2
    
    if max_or_mean == "max":
        cA = np.maximum(cA1, cA2)
        cH = np.maximum(cH1, cH2)
        cV = np.maximum(cV1, cV2)
        cD = np.maximum(cD1, cD2)
    if max_or_mean == "mean":
        cA = (cA1 + cA2) / 2
        cH = (cH1 + cH2) / 2
        cV = (cV1 + cV2) / 2
        cD = (cD1 + cD2) / 2
    if max_or_mean == "min":
        cA = np.minimum(cA1, cA2)
        cH = np.minimum(cH1, cH2)
        cV = np.minimum(cV1, cV2)
        cD = np.minimum(cD1, cD2)
    
    
    coeff = cA, (cH, cV, cD)
    
    return coeff, coeff1, coeff2, max_or_mean


os.chdir('C:\\Facultate\\Master\\SSIPAI\\Project\\new\\images')
image1 = rgb2gray(plt.imread('venice_normal.png'))
image2 = rgb2gray(plt.imread('venice_under.png'))
# image1 = plt.imread('lab1.jpg')
# image2 = plt.imread('lab2.jpg')
wavelet = 'haar'

coeff, coeff1, coeff2, method = wt_coeff(image1, image2, wavelet, "mean")

cA1, (cH1, cV1, cD1) = coeff1
cA2, (cH2, cV2, cD2) = coeff2

fused_image = pywt.idwt2(coeff, wavelet)

fig1 = plt.figure(figsize=(5, 7))
show(2, 2, 1, cA1, "cA1")
show(2, 2, 2, cH1, "cH1")
show(2, 2, 3, cV1, "cV1")
show(2, 2, 4, cD1, "cD1")
plt.tight_layout()


fig2 = plt.figure(figsize=(5, 7))
show(2, 2, 1, cA2, "cA2")
show(2, 2, 2, cH2, "cH2")
show(2, 2, 3, cV2, "cV2")
show(2, 2, 4, cD2, "cD2")
plt.tight_layout()


fig3 = plt.figure(figsize=(7, 7))
show(1, 1, 1, fused_image, f"Fused Image {method}")
plt.tight_layout()


fig4 = plt.figure(figsize = (10, 7))
show(1, 3, 1, image1, "Image 1")
show(1, 3, 2, image2, "Image 2")
show(1, 3, 3, fused_image, f"Fused Image {method}")
plt.tight_layout()


