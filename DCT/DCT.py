import copy
import numpy as np
import math
import struct
import time

B_size = 8
LEN_ROW = 512


def GetImage(f):
    image = np.empty((0, LEN_ROW))

    while True:

        data = f.read(LEN_ROW)
        if data == b"":
            break

        lst = []
        for i in range(LEN_ROW):

            lst.append(data[i])

        image = np.append(image, np.array([lst]), axis=0)

    
    return image

def DCT8by8(image):
    global B_size
    
    F = np.zeros((LEN_ROW, LEN_ROW))
    
    for i in range(0,LEN_ROW,B_size):
        for j in range(0,LEN_ROW,B_size):
            
            for u in range(B_size):
                for v in range(B_size):
                        
                        SUM_XY = 0
                        for x in range(B_size):
                            for y in range(B_size):
                                SUM_XY += (math.cos((2*y+1)*v*math.pi / (2*B_size))) * \
                                    (math.cos((2*x+1)*u*math.pi / (2*B_size))) * \
                                        image[x+i,y+j]
                        
                        F[i+u,j+v] = k(u)*k(v)*SUM_XY
            
    return F

def IDCT8by8(freq):
    global B_size
    image = np.zeros((LEN_ROW, LEN_ROW))
    for i in range(0,LEN_ROW,B_size):
        for j in range(0,LEN_ROW,B_size):
            
            for x in range(B_size):
                for y in range(B_size):
                    SUM_XY = 0
                    for u in range(B_size): 
                        for v in range(B_size):
                            SUM_XY += k(u) * k(v) * (math.cos((2*y+1)*v*math.pi / (2*B_size))) *\
                                (math.cos((2*x+1)*u*math.pi / (2*B_size))) * freq[u+i,v+j]
                            
                    image[x+i,y+j] = SUM_XY
    return image
    
def k(u):
    if u == 0:
        return math.sqrt(1/B_size)
    else:
        return math.sqrt(2/B_size)

def Stretching(array):
    # * Mapping to range in 0~255 - Normarlize
    
    flatten = np.ravel(array, order='C')
    flatten_list = sorted(flatten.tolist())
    min_value = flatten_list[0]
    max_value = flatten_list[-1]
    
    for row in range(0,len(array)):
        for col in range(0,len(array)):
            array[row,col] = ((array[row,col]-min_value)/(max_value-min_value))*255 #* new pixel
            #* Using Basic Contrast Stretching to cover negative value
    return array

def GetHeaderFromBMP(filename):
    # * Parsing header field to find offset field where image is located

    bmp = open(filename, "rb")
    
    print(bmp.read(10))  # * for reading raw_offset field in header
    raw_offset = (struct.unpack("i", bmp.read(4)))[0]

    print("raw_offset : ", hex(raw_offset))
    bmp.close()

    # * To save header from bmp file
    bmp = open(filename, "rb")
    header = bmp.read(raw_offset)
    bmp.close()

    return header

def MakeBMP(image,header,filename):
    freq_bmp = open(filename, "wb")
    freq_bmp.write(header)

     # * bmp file - input reversed image vertically
    for row in reversed(range(0,LEN_ROW)):
        for col in range(0,LEN_ROW):
            byte = int.to_bytes(round(image[row, col]), 1, "big")
            freq_bmp.write(byte)

    freq_bmp.close()

def MSE(imageA, imageB):
    global LEN_ROW

    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= (LEN_ROW ** 2)

    return err

def main():
    filename = "lena_bmp_512x512_new.bmp"
    try:
        header = GetHeaderFromBMP(filename)
    except:
        # * There is no BMP file to get header
        print("[-] There is no BMP file to extract BMP HEADER . . .")
        print("[-] So I already put the header. It is hard coding.")
        header = b'BM6\x04\x04\x00\x00\x00\x00\x006\x04\x00\x00(\x00\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x01\x00\x08\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x01\x01\x00\x02\x02\x02\x00\x03\x03\x03\x00\x04\x04\x04\x00\x05\x05\x05\x00\x06\x06\x06\x00\x07\x07\x07\x00\x08\x08\x08\x00\t\t\t\x00\n\n\n\x00\x0b\x0b\x0b\x00\x0c\x0c\x0c\x00\r\r\r\x00\x0e\x0e\x0e\x00\x0f\x0f\x0f\x00\x10\x10\x10\x00\x11\x11\x11\x00\x12\x12\x12\x00\x13\x13\x13\x00\x14\x14\x14\x00\x15\x15\x15\x00\x16\x16\x16\x00\x17\x17\x17\x00\x18\x18\x18\x00\x19\x19\x19\x00\x1a\x1a\x1a\x00\x1b\x1b\x1b\x00\x1c\x1c\x1c\x00\x1d\x1d\x1d\x00\x1e\x1e\x1e\x00\x1f\x1f\x1f\x00   \x00!!!\x00"""\x00###\x00$$$\x00%%%\x00&&&\x00\'\'\'\x00(((\x00)))\x00***\x00+++\x00,,,\x00---\x00...\x00///\x00000\x00111\x00222\x00333\x00444\x00555\x00666\x00777\x00888\x00999\x00:::\x00;;;\x00<<<\x00===\x00>>>\x00???\x00@@@\x00AAA\x00BBB\x00CCC\x00DDD\x00EEE\x00FFF\x00GGG\x00HHH\x00III\x00JJJ\x00KKK\x00LLL\x00MMM\x00NNN\x00OOO\x00PPP\x00QQQ\x00RRR\x00SSS\x00TTT\x00UUU\x00VVV\x00WWW\x00XXX\x00YYY\x00ZZZ\x00[[[\x00\\\\\\\x00]]]\x00^^^\x00___\x00```\x00aaa\x00bbb\x00ccc\x00ddd\x00eee\x00fff\x00ggg\x00hhh\x00iii\x00jjj\x00kkk\x00lll\x00mmm\x00nnn\x00ooo\x00ppp\x00qqq\x00rrr\x00sss\x00ttt\x00uuu\x00vvv\x00www\x00xxx\x00yyy\x00zzz\x00{{{\x00|||\x00}}}\x00~~~\x00\x7f\x7f\x7f\x00\x80\x80\x80\x00\x81\x81\x81\x00\x82\x82\x82\x00\x83\x83\x83\x00\x84\x84\x84\x00\x85\x85\x85\x00\x86\x86\x86\x00\x87\x87\x87\x00\x88\x88\x88\x00\x89\x89\x89\x00\x8a\x8a\x8a\x00\x8b\x8b\x8b\x00\x8c\x8c\x8c\x00\x8d\x8d\x8d\x00\x8e\x8e\x8e\x00\x8f\x8f\x8f\x00\x90\x90\x90\x00\x91\x91\x91\x00\x92\x92\x92\x00\x93\x93\x93\x00\x94\x94\x94\x00\x95\x95\x95\x00\x96\x96\x96\x00\x97\x97\x97\x00\x98\x98\x98\x00\x99\x99\x99\x00\x9a\x9a\x9a\x00\x9b\x9b\x9b\x00\x9c\x9c\x9c\x00\x9d\x9d\x9d\x00\x9e\x9e\x9e\x00\x9f\x9f\x9f\x00\xa0\xa0\xa0\x00\xa1\xa1\xa1\x00\xa2\xa2\xa2\x00\xa3\xa3\xa3\x00\xa4\xa4\xa4\x00\xa5\xa5\xa5\x00\xa6\xa6\xa6\x00\xa7\xa7\xa7\x00\xa8\xa8\xa8\x00\xa9\xa9\xa9\x00\xaa\xaa\xaa\x00\xab\xab\xab\x00\xac\xac\xac\x00\xad\xad\xad\x00\xae\xae\xae\x00\xaf\xaf\xaf\x00\xb0\xb0\xb0\x00\xb1\xb1\xb1\x00\xb2\xb2\xb2\x00\xb3\xb3\xb3\x00\xb4\xb4\xb4\x00\xb5\xb5\xb5\x00\xb6\xb6\xb6\x00\xb7\xb7\xb7\x00\xb8\xb8\xb8\x00\xb9\xb9\xb9\x00\xba\xba\xba\x00\xbb\xbb\xbb\x00\xbc\xbc\xbc\x00\xbd\xbd\xbd\x00\xbe\xbe\xbe\x00\xbf\xbf\xbf\x00\xc0\xc0\xc0\x00\xc1\xc1\xc1\x00\xc2\xc2\xc2\x00\xc3\xc3\xc3\x00\xc4\xc4\xc4\x00\xc5\xc5\xc5\x00\xc6\xc6\xc6\x00\xc7\xc7\xc7\x00\xc8\xc8\xc8\x00\xc9\xc9\xc9\x00\xca\xca\xca\x00\xcb\xcb\xcb\x00\xcc\xcc\xcc\x00\xcd\xcd\xcd\x00\xce\xce\xce\x00\xcf\xcf\xcf\x00\xd0\xd0\xd0\x00\xd1\xd1\xd1\x00\xd2\xd2\xd2\x00\xd3\xd3\xd3\x00\xd4\xd4\xd4\x00\xd5\xd5\xd5\x00\xd6\xd6\xd6\x00\xd7\xd7\xd7\x00\xd8\xd8\xd8\x00\xd9\xd9\xd9\x00\xda\xda\xda\x00\xdb\xdb\xdb\x00\xdc\xdc\xdc\x00\xdd\xdd\xdd\x00\xde\xde\xde\x00\xdf\xdf\xdf\x00\xe0\xe0\xe0\x00\xe1\xe1\xe1\x00\xe2\xe2\xe2\x00\xe3\xe3\xe3\x00\xe4\xe4\xe4\x00\xe5\xe5\xe5\x00\xe6\xe6\xe6\x00\xe7\xe7\xe7\x00\xe8\xe8\xe8\x00\xe9\xe9\xe9\x00\xea\xea\xea\x00\xeb\xeb\xeb\x00\xec\xec\xec\x00\xed\xed\xed\x00\xee\xee\xee\x00\xef\xef\xef\x00\xf0\xf0\xf0\x00\xf1\xf1\xf1\x00\xf2\xf2\xf2\x00\xf3\xf3\xf3\x00\xf4\xf4\xf4\x00\xf5\xf5\xf5\x00\xf6\xf6\xf6\x00\xf7\xf7\xf7\x00\xf8\xf8\xf8\x00\xf9\xf9\xf9\x00\xfa\xfa\xfa\x00\xfb\xfb\xfb\x00\xfc\xfc\xfc\x00\xfd\xfd\xfd\x00\xfe\xfe\xfe\x00\xff\xff\xff\x00'

    f = open("lena_raw_512x512.raw", "rb")
    lena_image = GetImage(f)
    f.close()
    MakeBMP(lena_image,header,"lena_original.bmp")

    print("[*] LENA : DCT . . .")
    Frequency_image_lena = DCT8by8(lena_image)
    
    #* for plot spectrum
    F_array_lena = copy.deepcopy(Frequency_image_lena)
    F_array_lena = Stretching(F_array_lena)
    
    filename = "lena_freq.bmp"
    MakeBMP(F_array_lena,header,filename)

    print("[*] LENA : Inverse DCT . . .")
    lena_arr = IDCT8by8(Frequency_image_lena)

    filename = "lena.bmp"
    MakeBMP(lena_arr,header,filename)
    
    f2 = open("BOAT512.raw.raw","rb")
    boat_image = GetImage(f2)
    f2.close()
    MakeBMP(boat_image,header,"boat_original.bmp")
    print("[*] BOAT : DCT . . .")
    Frequency_image_boat = DCT8by8(boat_image)

    F_array_boat = copy.deepcopy(Frequency_image_boat)
    F_array_boat = Stretching(F_array_boat)
    
    filename= "boat_freq.bmp"
    MakeBMP(F_array_boat,header,filename)
    print("[*] BOAT : Inverse DCT . . .")
    boat_arr = IDCT8by8(Frequency_image_boat)
    filename = "boat.bmp"
    MakeBMP(boat_arr,header,filename)

    #* Compute the MSE - original, DCT/IDCT
    print(f"[*] LENA - MSE {MSE(lena_image,lena_arr)}")
    print(f"[*] BOAT - MSE {MSE(boat_image,boat_arr)}")


if __name__ == "__main__":
    main()