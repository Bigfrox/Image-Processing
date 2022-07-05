from threading import Thread
import numpy as np
import math
import copy
import time
import struct
import random


#* n0 : the number of edge that exceeds threshold=150
#* n1 : number of edge(missing or new)

def MakeGaussianNoise(input_image,header,original,SNR):
    
    global LEN_ROW
    print("Gaussian")
    #* Gaussian Noise 
    variance = np.var(input_image)
    
    stddev_noise = variance/pow(10.0,SNR/10)
    #* input image, noise image are array!

    noise_img = AddGaussianNoise(input_image, stddev_noise)
    print(noise_img.shape)
    noise_file = open(original+"_noise.bmp","wb")
    
    noise_file.write(header)

     # * bmp file - input reversed image vertically
    print("LEN_ROW",LEN_ROW)
    for row in reversed(range(0,LEN_ROW)):
        for col in range(0,LEN_ROW):
            pixel = noise_img[row, col]
            if pixel > 255:
                pixel = 255
            if pixel < 0:
                pixel = 0

            byte = int.to_bytes(int(pixel), 1, "big")
            noise_file.write(byte)

    noise_file.close()

    return noise_img


def AddGaussianNoise(image, noise_var):
    row, col = image.shape
    mean = 0
    sigma = math.sqrt(noise_var)
    gauss = np.random.normal(mean,sigma, (row,col))
    gauss = gauss.reshape(row,col)
    noisy = image + gauss
    return noisy

def MSE(imageA, imageB):
    global LEN_ROW

    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= (LEN_ROW ** 2)

    return err

def Roberts_Mask(image, roberts_arr, header, isNoisy):

    if isNoisy:
        filename = "roberts_lena_noisy"
    else:
        filename = "roberts_lena_origin"

    roberts_mask_x = np.array([[0, 0, -1], [0, 1, 0], [0, 0, 0]])
    roberts_mask_y = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]])

    for j in range(1, LEN_ROW - 1):
        for i in range(1, LEN_ROW - 1):

            target = np.array(
                [
                    [image[i - 1, j - 1], image[i, j - 1], image[i + 1, j - 1]],
                    [image[i - 1, j], image[i, j], image[i + 1, j]],
                    [image[i - 1, j + 1], image[i, j + 1], image[i + 1, j + 1]],
                ]
            )

            gx = target * roberts_mask_x
            gy = target * roberts_mask_y
            magnitude = math.sqrt(pow(gx.sum(), 2) + pow(gy.sum(), 2))
            if magnitude > 255:
                magnitude = 255
            if magnitude < 0:
                magnitude = 0
            
            roberts_arr[i, j] = magnitude
    
    # * write header to new bmp file
    #roberts = open("roberts.bmp", "wb")
    roberts = open(filename+".bmp", "wb")
    roberts.write(header)

    # * bmp file - input reversed image vertically
    #! 5
    row = LEN_ROW - 1
    for col in range(LEN_ROW):
        roberts_arr[LEN_ROW - 1, col] = 0
        byte = int.to_bytes(int(roberts_arr[LEN_ROW - 1, col]), 1, "big")
        #byte = int.to_bytes(0, 1, "big")
        roberts.write(byte)

    for row in reversed(range(1, LEN_ROW - 1)):
        #! 2 - reversed
        roberts_arr[row, 0] = 0
        byte = int.to_bytes(int(roberts_arr[row, 0]), 1, "big")
        #byte = int.to_bytes(0, 1, "big")
        roberts.write(byte)

        #! 3 - reversed
        for col in range(1, LEN_ROW - 1):
            byte = int.to_bytes(int(roberts_arr[row, col]), 1, "big")
            roberts.write(byte)

        #! 4 - reversed
        roberts_arr[row, LEN_ROW - 1] = 0
        byte = int.to_bytes(int(roberts_arr[row, LEN_ROW - 1]), 1, "big")
        #byte = int.to_bytes(0, 1, "big")
        roberts.write(byte)

    #! 1
    for col in range(LEN_ROW):
        roberts_arr[0, col] = 0
        byte = int.to_bytes(int(roberts_arr[0, col]), 1, "big")
        #byte = int.to_bytes(0, 1, "big")
        roberts.write(byte)

    roberts.close()
    return roberts_arr

def Sobel_Mask(image, sobel_arr, header, isNoisy):

    if isNoisy:
        filename = "sobel_lena_noisy"
    else:
        filename = "sobel_lena_origin"

    sobel_mask_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_mask_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    for j in range(1, LEN_ROW - 1):
        for i in range(1, LEN_ROW - 1):

            target = np.array(
                [
                    [image[i - 1, j - 1], image[i, j - 1], image[i + 1, j - 1]],
                    [image[i - 1, j], image[i, j], image[i + 1, j]],
                    [image[i - 1, j + 1], image[i, j + 1], image[i + 1, j + 1]],
                ]
            )

            gx = target * sobel_mask_x / 4
            gy = target * sobel_mask_y / 4
            magnitude = math.sqrt(pow(gx.sum(), 2) + pow(gy.sum(), 2))
            if magnitude > 255:
                magnitude = 255
            if magnitude < 0:
                magnitude = 0
            
            sobel_arr[i, j] = magnitude
            

    # * write header to new bmp file
    sobel = open(filename+".bmp", "wb")
    sobel.write(header)

    # * bmp file - input reversed image vertically
    #! 5
    row = LEN_ROW - 1
    for col in range(LEN_ROW):
        #* byte = int.to_bytes(int(sobel_arr[LEN_ROW - 1, col]), 1, "big")
        sobel_arr[LEN_ROW - 1, col] = 0
        byte = int.to_bytes(int(sobel_arr[LEN_ROW - 1, col]), 1, "big")
        #byte = int.to_bytes(0, 1, "big")
        sobel.write(byte)

    for row in reversed(range(1, LEN_ROW - 1)):
        #! 2 - reversed
        #* byte = int.to_bytes(int(sobel_arr[row, 0]), 1, "big")
        sobel_arr[row, 0] = 0
        byte = int.to_bytes(int(sobel_arr[row, 0]), 1, "big")
        #byte = int.to_bytes(0, 1, "big")
        sobel.write(byte)

        #! 3 - reversed
        for col in range(1, LEN_ROW - 1):
            byte = int.to_bytes(int(sobel_arr[row, col]), 1, "big")
            sobel.write(byte)

        #! 4 - reversed
        #* byte = int.to_bytes(int(sobel_arr[row, LEN_ROW - 1]), 1, "big")
        
        sobel_arr[row, LEN_ROW - 1] = 0
        byte = int.to_bytes(int(sobel_arr[row, LEN_ROW - 1]), 1, "big")
        #byte = int.to_bytes(0, 1, "big")
        sobel.write(byte)

    #! 1
    for col in range(LEN_ROW):
        #* byte = int.to_bytes(int(sobel_arr[0, col]), 1, "big")
        sobel_arr[0, col] = 0
        byte = int.to_bytes(int(sobel_arr[0, col]), 1, "big")
        #byte = int.to_bytes(0, 1, "big")
        sobel.write(byte)

    sobel.close()
    return sobel_arr

def Prewit_Mask(image,prewit_arr,header, isNoisy):
    if isNoisy:
        filename = "prewit_lena_noisy"
    else:
        filename = "prewit_lena_origin"

    prewit_mask_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewit_mask_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    for j in range(1, LEN_ROW - 1):
        for i in range(1, LEN_ROW - 1):

            target = np.array(
                [
                    [image[i - 1, j - 1], image[i, j - 1], image[i + 1, j - 1]],
                    [image[i - 1, j], image[i, j], image[i + 1, j]],
                    [image[i - 1, j + 1], image[i, j + 1], image[i + 1, j + 1]],
                ]
            )

            gx = target * prewit_mask_x / 3
            gy = target * prewit_mask_y / 3
            magnitude = math.sqrt(pow(gx.sum(), 2) + pow(gy.sum(), 2))
            if magnitude > 255:
                magnitude = 255
            if magnitude < 0:
                magnitude = 0
            
            prewit_arr[i, j] = magnitude


    # * write header to new bmp file
    prewit = open(filename+".bmp", "wb")
    prewit.write(header)

    # * bmp file - input reversed image vertically
    #! 5
    row = LEN_ROW - 1
    for col in range(LEN_ROW):
        prewit_arr[LEN_ROW - 1, col] = 0
        byte = int.to_bytes(int(prewit_arr[LEN_ROW - 1, col]), 1, "big")
        #byte = int.to_bytes(0, 1, "big")
        prewit.write(byte)

    for row in reversed(range(1, LEN_ROW - 1)):
        #! 2 - reversed
        prewit_arr[row, 0] = 0
        byte = int.to_bytes(int(prewit_arr[row, 0]), 1, "big")
        #byte = int.to_bytes(0, 1, "big")
        prewit.write(byte)

        #! 3 - reversed
        for col in range(1, LEN_ROW - 1):
            # print(f"[{row},{col}] : {hex(int(arr[row,col]))} => {hex(int(prewit_arr[row,col]))}")
            byte = int.to_bytes(int(prewit_arr[row, col]), 1, "big")
            prewit.write(byte)

        #! 4 - reversed
        prewit_arr[row, LEN_ROW - 1] = 0
        byte = int.to_bytes(int(prewit_arr[row, LEN_ROW - 1]), 1, "big")
        #byte = int.to_bytes(0, 1, "big")
        prewit.write(byte)

    #! 1
    for col in range(LEN_ROW):
        prewit_arr[0, col] = 0
        byte = int.to_bytes(int(prewit_arr[0, col]), 1, "big")
        #byte = int.to_bytes(0, 1, "big")
        prewit.write(byte)

    prewit.close()
    return prewit_arr

def Stochastic_Mask(image, stochastic_arr, header, isNoisy):
    if isNoisy:
        filename = "stochastic_lena_noisy"
    else:
        filename = "stochastic_lena_origin"
    
    # * Stochastic gradient 5x5 matrix
    stochastic_mask_x = np.array(
        [
            [0.267, 0.364, 0, -0.364, -0.267],
            [0.373, 0.562, 0, -0.562, -0.373],
            [0.463, 1.0, 0, -1.0, -0.463],
            [0.373, 0.562, 0, -0.562, -0.373],
            [0.267, 0.364, 0, -0.364, -0.267],
        ]
    )
    #print(stochastic_mask_x.shape)
    stochastic_mask_y = stochastic_mask_x.T
    #print(stochastic_mask_y)

    for j in range(2, LEN_ROW - 2):
        for i in range(2, LEN_ROW - 2):

            # * 5x5 matrix
            target = np.array(
                [
                    [
                        image[i - 2, j - 2],
                        image[i - 1, j - 2],
                        image[i, j - 2],
                        image[i + 1, j - 2],
                        image[i + 2, j - 2],
                    ],
                    [
                        image[i - 2, j - 1],
                        image[i - 1, j - 1],
                        image[i, j - 1],
                        image[i + 1, j - 1],
                        image[i + 2, j - 1],
                    ],
                    [
                        image[i - 2, j],
                        image[i - 1, j],
                        image[i, j],
                        image[i + 1, j],
                        image[i + 2, j],
                    ],
                    [
                        image[i - 2, j + 1],
                        image[i - 1, j + 1],
                        image[i, j + 1],
                        image[i + 1, j + 1],
                        image[i + 2, j + 1],
                    ],
                    [
                        image[i - 2, j + 2],
                        image[i - 1, j + 2],
                        image[i, j + 2],
                        image[i + 1, j + 2],
                        image[i + 2, j + 2],
                    ],
                ]
            )
            
            gx = target * stochastic_mask_x
            gy = target * stochastic_mask_y
            magnitude = math.sqrt(pow(gx.sum(), 2) + pow(gy.sum(), 2))
            
            
            stochastic_arr[i, j] = magnitude
            
    
            
    # * Mapping to range in 0~255 - Normarlize
    print(stochastic_arr.shape)
    stochastic_flatten = np.ravel(stochastic_arr, order='C')
    stochastic_flatten_list = sorted(stochastic_flatten.tolist())
    min_stochastic = stochastic_flatten_list[0]
    max_stochastic = stochastic_flatten_list[-1] 
    #stochastic_arr_new = copy.deepcopy(stochastic_arr)
    for row in range(2,LEN_ROW-2):
        for col in range(2,LEN_ROW-2):
            stochastic_arr[col,row] = ((stochastic_arr[col,row]-min_stochastic)/(max_stochastic-min_stochastic))*255 #* new pixel
            #* Using Basic Contrast Stretching to cover negative value
    
    


    # * write header to new bmp file
    stochastic = open(filename+".bmp", "wb")
    stochastic.write(header)

    # * bmp file - input reversed image vertically
    # ? zero-padding : option
    #! 5
    row = LEN_ROW - 1
    for col in range(LEN_ROW):
        stochastic_arr[LEN_ROW - 1, col] = 0
        byte = int.to_bytes(int(stochastic_arr[LEN_ROW - 1, col]), 1, "big")
        #byte = int.to_bytes(0, 1, "big")
        stochastic.write(byte)

    row = LEN_ROW - 2
    for col in range(LEN_ROW):
        stochastic_arr[LEN_ROW - 2, col] = 0
        byte = int.to_bytes(int(stochastic_arr[LEN_ROW - 2, col]), 1, "big")
        #byte = int.to_bytes(0, 1, "big")
        stochastic.write(byte)

    for row in reversed(range(2, LEN_ROW - 2)):
        #! 2 - reversed
        stochastic_arr[row, 0] = 0
        byte = int.to_bytes(int(stochastic_arr[row, 0]), 1, "big")
        #byte = int.to_bytes(0, 1, "big")
        stochastic.write(byte)

        stochastic_arr[row, 1] = 0
        byte = int.to_bytes(int(stochastic_arr[row, 1]), 1, "big")
        #byte = int.to_bytes(0, 1, "big")
        stochastic.write(byte)

        #! 3 - reversed
        for col in range(2, LEN_ROW - 2):

            # print(f"[{row},{col}] : {hex(int(arr[row,col]))} => {hex(int(stochastic_arr[row,col]))}")
            byte = int.to_bytes(int(stochastic_arr[row, col]), 1, "big")
            stochastic.write(byte)

        #! 4 - reversed
        stochastic_arr[row, LEN_ROW - 2] = 0
        byte = int.to_bytes(int(stochastic_arr[row, LEN_ROW - 2]), 1, "big")
        #byte = int.to_bytes(0, 1, "big")
        stochastic.write(byte)

        stochastic_arr[row, LEN_ROW - 1] = 0
        byte = int.to_bytes(int(stochastic_arr[row, LEN_ROW - 1]), 1, "big")
        #byte = int.to_bytes(0, 1, "big")
        stochastic.write(byte)

    #! 1
    for col in range(LEN_ROW):
        stochastic_arr[0, col] = 0
        byte = int.to_bytes(int(stochastic_arr[0, col]), 1, "big")
        #byte = int.to_bytes(0, 1, "big")
        stochastic.write(byte)

    for col in range(LEN_ROW):
        stochastic_arr[1, col] = 0
        byte = int.to_bytes(int(stochastic_arr[1, col]), 1, "big")
        #byte = int.to_bytes(0, 1, "big")
        stochastic.write(byte)
    
    stochastic.close()
    return stochastic_arr

def Get_Error_rate(origin_image, noisy_image, THRESHOLD):
    
    edge_original = copy.deepcopy(origin_image)
    edge_noisy = copy.deepcopy(noisy_image)

    n0 = 0
    n1 = 0
    for row in range(LEN_ROW):
        for col in range(LEN_ROW):
            if edge_original[col,row] > THRESHOLD:
                edge_original[col,row] = 255
                n0 += 1
            else:
                edge_original[col,row] = 0

    for row in range(LEN_ROW):
        for col in range(LEN_ROW):
            if edge_noisy[col,row] > THRESHOLD:
                edge_noisy[col,row] = 255
            else:
                edge_noisy[col,row] = 0
    
    for row in range(LEN_ROW):
        for col in range(LEN_ROW):
            if edge_original[col,row] != edge_noisy[col,row]:
                n1 += 1

    #* Error Rate : P_e
    print("n0:",n0)
    print("n1:",n1)
    error_rate = float(n1)/float(n0)
    
    return error_rate
    

def main():
    
    THRESHOLD = 150 #! Threshold =150
    global LEN_ROW
    LEN_ROW = 512

    # * Parsing header field to find offset field where image is located

    bmp = open("lena_bmp_512x512_new.bmp", "rb")
    
    print(bmp.read(10))  # * for reading raw_offset field in header
    raw_offset = (struct.unpack("i", bmp.read(4)))[0]

    print("raw_offset : ", hex(raw_offset))
    bmp.close()

    # * To save header from bmp file
    bmp = open("lena_bmp_512x512_new.bmp", "rb")
    header = bmp.read(raw_offset)
    # raw = bmp.read(512*512)
    bmp.close()

    # * Problem 1.
    f = open("lena_raw_512x512.raw", "rb")    
    lena_arr = np.empty((0, LEN_ROW))

    while True:

        data = f.read(LEN_ROW)
        if data == b"":
            break

        lst = []
        for i in range(LEN_ROW):

            lst.append(data[i])

        lena_arr = np.append(lena_arr, np.array([lst]), axis=0)

    f.close()

    noisy_lena_arr = MakeGaussianNoise(lena_arr,header,"lena",SNR=8)
    


    #* Roberts
    origin_roberts_arr = copy.deepcopy(lena_arr)
    origin_roberts_arr = Roberts_Mask(lena_arr, origin_roberts_arr, header, isNoisy=False)
    noisy_roberts_arr = copy.deepcopy(noisy_lena_arr)
    noisy_roberts_arr = Roberts_Mask(noisy_lena_arr, noisy_roberts_arr, header, isNoisy=True)
    

    # * Sobel Operator
    print("Sobel")
    origin_sobel_arr = copy.deepcopy(lena_arr)
    origin_sobel_arr = Sobel_Mask(lena_arr,origin_sobel_arr,header, isNoisy=False)
    noisy_sobel_arr = copy.deepcopy(noisy_lena_arr)
    noisy_sobel_arr = Sobel_Mask(noisy_lena_arr,noisy_sobel_arr,header, isNoisy=True)

    # * Prewit
    print("Prewit")
    origin_prewit_arr = copy.deepcopy(lena_arr)
    origin_prewit_arr = Prewit_Mask(lena_arr,origin_prewit_arr,header, isNoisy=False)
    noisy_prewit_arr = copy.deepcopy(noisy_lena_arr)
    noisy_prewit_arr = Prewit_Mask(noisy_lena_arr,noisy_prewit_arr,header, isNoisy=True)
    

    print("Stochastic")
    origin_stochastic_arr = copy.deepcopy(lena_arr)
    origin_stochastic_arr = Stochastic_Mask(lena_arr,origin_stochastic_arr,header, isNoisy=False)
    noisy_stochastic_arr = copy.deepcopy(noisy_lena_arr)
    noisy_stochastic_arr = Stochastic_Mask(noisy_lena_arr,noisy_stochastic_arr,header, isNoisy=True)
    
    #* edge detection
    
    error_rate = Get_Error_rate(origin_image= origin_roberts_arr, noisy_image=noisy_roberts_arr,THRESHOLD=THRESHOLD)
    print(f"[Roberts] Error Rate P_e : {error_rate}")
    error_rate = Get_Error_rate(origin_image= origin_sobel_arr, noisy_image=noisy_sobel_arr,THRESHOLD=THRESHOLD)
    print(f"[Sobel] Error Rate P_e : {error_rate}")
    error_rate = Get_Error_rate(origin_image= origin_prewit_arr, noisy_image=noisy_prewit_arr,THRESHOLD=THRESHOLD)
    print(f"[Prewit] Error Rate P_e : {error_rate}")
    error_rate = Get_Error_rate(origin_image= origin_stochastic_arr, noisy_image=noisy_stochastic_arr,THRESHOLD=THRESHOLD)
    print(f"[Stochastic] Error Rate P_e : {error_rate}")




    # * Problem 2.
    """
    Compare the performance between the 3x3 Low-pass and Median filters for a noisy image with SNR=9dB. 
    For an objective comparison, obtain the MSE (mean square error) for each result. 
    For this assignment, use 512x512 grayscale image of BOAT.raw.  
    """

    #* Make bmp file using raw file
    boat_raw = open("BOAT512.raw.raw", "rb")
    
    boat_arr = np.empty((0, LEN_ROW))

    while True:

        data = boat_raw.read(LEN_ROW)
        if data == b"":
            break

        lst = []
        for i in range(LEN_ROW):
            lst.append(data[i])

        boat_arr = np.append(boat_arr, np.array([lst]), axis=0)

    boat_raw.close()

    print("boat_bmp")
    boat_bmp = open("boat.bmp", "wb")
    boat_bmp.write(header)

     # * bmp file - input reversed image vertically

    for row in reversed(range(0,LEN_ROW)):
        for col in range(0,LEN_ROW):
            byte = int.to_bytes(int(boat_arr[row, col]), 1, "big")
            boat_bmp.write(byte)

    boat_bmp.close()


    noisy_boat_arr = MakeGaussianNoise(boat_arr,header,"boat",SNR=9)

    #* Median filter
    print("Median filter")
    boat_median_arr = copy.deepcopy(noisy_boat_arr)

    for j in range(1, LEN_ROW - 1):
        for i in range(1, LEN_ROW - 1):

            target = np.array(
                [
                    [boat_arr[i - 1, j - 1], boat_arr[i, j - 1], boat_arr[i + 1, j - 1]],
                    [boat_arr[i - 1, j], boat_arr[i, j], boat_arr[i + 1, j]],
                    [boat_arr[i - 1, j + 1], boat_arr[i, j + 1], boat_arr[i + 1, j + 1]],
                ]
            )
            
            target_flatten = np.ravel(target, order='C')
            target_lst = target_flatten.tolist()
            sorted_target = sorted(target_lst)
            median_index = int(len(sorted_target) / 2)
            
            magnitude = sorted_target[median_index]
            boat_median_arr[i, j] = magnitude

    print("median bmp")
    boat_median = open("boat_median.bmp","wb")
    boat_median.write(header)

    # * bmp file - input reversed image vertically
    #! 5
    row = LEN_ROW - 1
    for col in range(LEN_ROW):
        #* byte = int.to_bytes(int(boat_median_arr[LEN_ROW - 1, col]), 1, "big")
        byte = int.to_bytes(0, 1, "big")
        boat_median.write(byte)

    for row in reversed(range(1,LEN_ROW-1)):
        #! 2
        #* byte = int.to_bytes(int(boat_median_arr[row, 0]), 1, "big")
        byte = int.to_bytes(0, 1, "big")
        boat_median.write(byte)

        #! 3
        for col in range(1,LEN_ROW-1):
            byte = int.to_bytes(int(boat_median_arr[row, col]), 1, "big")
            boat_median.write(byte)

        #! 4 - reversed
        #* byte = int.to_bytes(int(boat_median_arr[row, LEN_ROW - 1]), 1, "big")
        byte = int.to_bytes(0, 1, "big")
        boat_median.write(byte)
    
    #! 1
    for col in range(LEN_ROW):
        #* byte = int.to_bytes(int(boat_median_arr[0, col]), 1, "big")
        byte = int.to_bytes(0, 1, "big")
        boat_median.write(byte)
    
    boat_median.close()



    # * Low pass filter - Blurring, 3x3 Mask
    print("Low pass filter")

    
    lowpass_arr = copy.deepcopy(noisy_boat_arr)
    lowpass_mask = np.array([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]])
    

    for j in range(1, LEN_ROW - 1):
        for i in range(1, LEN_ROW - 1):

            target = np.array(
                [
                    [boat_arr[i - 1, j - 1], boat_arr[i, j - 1], boat_arr[i + 1, j - 1]],
                    [boat_arr[i - 1, j], boat_arr[i, j], boat_arr[i + 1, j]],
                    [boat_arr[i - 1, j + 1], boat_arr[i, j + 1], boat_arr[i + 1, j + 1]],
                ]
            )

            gx = target * lowpass_mask
            
            magnitude = gx.sum()
            lowpass_arr[i,j] = magnitude
            

    # * write header to new bmp file
    lowpass_bmp = open("boat_lowpass.bmp", "wb")
    lowpass_bmp.write(header)

    # * bmp file - input reversed image vertically
    #! 5
    row = LEN_ROW - 1
    for col in range(LEN_ROW):
        #* byte = int.to_bytes(int(lowpass_arr[LEN_ROW - 1, col]), 1, "big")
        byte = int.to_bytes(0, 1, "big")
        lowpass_bmp.write(byte)

    for row in reversed(range(1, LEN_ROW - 1)):
        #! 2 - reversed
        #* byte = int.to_bytes(int(lowpass_arr[row, 0]), 1, "big")
        byte = int.to_bytes(0, 1, "big")
        lowpass_bmp.write(byte)

        #! 3 - reversed
        for col in range(1, LEN_ROW - 1):
            byte = int.to_bytes(int(lowpass_arr[row, col]), 1, "big")
            lowpass_bmp.write(byte)

        #! 4 - reversed
        #* byte = int.to_bytes(int(lowpass_arr[row, LEN_ROW - 1]), 1, "big")
        byte = int.to_bytes(0, 1, "big")
        lowpass_bmp.write(byte)

    #! 1
    for col in range(LEN_ROW):
        #* byte = int.to_bytes(int(lowpass_arr[0, col]), 1, "big")
        byte = int.to_bytes(0, 1, "big")
        lowpass_bmp.write(byte)

    lowpass_bmp.close()

    # * Compare MSE for each result with original image.
    print(f"MSE with Median Filter : {MSE(boat_arr,boat_median_arr)}")
    print(f"MSE with Low-Pass Filter : {MSE(boat_arr,lowpass_arr)}")

if __name__ == "__main__":
    main()