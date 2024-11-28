import os
import pywt
import numpy as np
import matplotlib.pyplot as plt

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def show(row, column, img_num, image, title):
    plt.subplot(row, column, img_num)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title(title)
    
def wt_coeff(image1, image2, wavelet, max_or_mean):
    
    coeff1_R = pywt.dwt2(image1[:, :, 0], wavelet)
    coeff1_G = pywt.dwt2(image1[:, :, 1], wavelet)
    coeff1_B = pywt.dwt2(image1[:, :, 2], wavelet)

    coeff2_R = pywt.dwt2(image2[:, :, 0], wavelet)
    coeff2_G = pywt.dwt2(image2[:, :, 1], wavelet)
    coeff2_B = pywt.dwt2(image2[:, :, 2], wavelet)

    cA1R, (cH1R, cV1R, cD1R) = coeff1_R
    cA1G, (cH1G, cV1G, cD1G) = coeff1_G
    cA1B, (cH1B, cV1B, cD1B) = coeff1_B

    cA2R, (cH2R, cV2R, cD2R) = coeff2_R
    cA2G, (cH2G, cV2G, cD2G) = coeff2_G
    cA2B, (cH2B, cV2B, cD2B) = coeff2_B

    if max_or_mean == "max":
        cAR = np.maximum(cA1R, cA2R) / 255
        cAG = np.maximum(cA1G, cA2G) / 255
        cAB = np.maximum(cA1B, cA2B) / 255
    
        cHR = np.maximum(cH1R, cH2R) / 255
        cHG = np.maximum(cH1G, cH2G) / 255
        cHB = np.maximum(cH1B, cH2B) / 255
    
        cVR = np.maximum(cV1R, cV2R) / 255
        cVG = np.maximum(cV1G, cV2G) / 255
        cVB = np.maximum(cV1B, cV2B) / 255
    
        cDR = np.maximum(cD1R, cD2R) / 255
        cDG = np.maximum(cD1G, cD2G) / 255
        cDB = np.maximum(cD1B, cD2B) / 255
    if max_or_mean == "mean":
        cAR = (cA1R + cA2R) / (255 * 2)
        cAG = (cA1G + cA2G) / (255 * 2)
        cAB = (cA1B + cA2B) / (255 * 2)
    
        cHR = (cH1R + cH2R) / (255 * 2)
        cHG = (cH1G + cH2G) / (255 * 2)
        cHB = (cH1B + cH2B) / (255 * 2)
    
        cVR = (cV1R + cV2R) / (255 * 2)
        cVG = (cV1G + cV2G) / (255 * 2)
        cVB = (cV1B + cV2B) / (255 * 2)
    
        cDR = (cD1R + cD2R) / (255 * 2)
        cDG = (cD1G + cD2G) / (255 * 2)
        cDB = (cD1B + cD2B) / (255 * 2)
    
    coeff_R = cAR, (cHR, cVR, cDR)
    coeff_G = cAG, (cHG, cVG, cDG)
    coeff_B = cAB, (cHB, cVB, cDB)
    
    return coeff_R, coeff_G, coeff_B
    
os.chdir('C:\\Facultate\\Master\\SSIPAI\\Project\\new')

image1 = plt.imread('bol_dark2.jpg')
image2 = plt.imread('bol_bright2.jpg')
wavelet = 'haar'

coeff_R, coeff_G, coeff_B = wt_coeff(image1, image2, wavelet, "mean")

cAR, (cHR, cVR, cDR) = coeff_R
cAG, (cHG, cVG, cDG) = coeff_G
cAB, (cHB, cVB, cDB) = coeff_B

reconstruct_R = pywt.idwt2((cAR, (cHR, cVR, cDR)), wavelet)
reconstruct_G = pywt.idwt2((cAG, (cHG, cVG, cDG)), wavelet)
reconstruct_B = pywt.idwt2((cAB, (cHB, cVB, cDB)), wavelet)



fused_image = np.stack((reconstruct_R, reconstruct_G, reconstruct_B), axis = -1)


fig = plt.figure(figsize = (10, 7))
show(1, 3, 1, image1, "Image 1")
show(1, 3, 2, image2, "Image 2")
show(1, 3, 3, fused_image, "Fused Image")
plt.tight_layout()


