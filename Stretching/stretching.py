import matplotlib.pyplot as plt



def main():
    
    hist = [0 for _ in range(256)]
    f = open('lena_bmp_512x512_new.bmp', 'rb')

    header = f.read(0x430)
    data = f.read(1)
    print(hex(data[0]))

    while data != b"":
        #print(data[0])
        hist[data[0]] += 1
        data = f.read(1)

    print(hist)
    print("the number of all pixel : ",sum(hist))

    #* ###
    #* Get Lowest value and Highest value of the original image
    
    for i_ in hist:
        if i_ > 0:
            min_original = hist.index(i_)
            break
    
    rev_hist = list(reversed(hist))
    print(rev_hist)

    for i__ in rev_hist:
        if i__ > 0:
            max_original = 255 - rev_hist.index(i__)
            break
    print("min : ",min_original)
    print("max : ",max_original)
    

    

    x = range(len(hist))
    plt.bar(x,hist,width=1,color='green',alpha=0.7,linewidth=4)

    plt.title('(a) Histogram', fontsize=12)

    plt.xlabel('Pixel Intersity', fontsize=10)

    plt.ylabel('frequency', fontsize=10)


    plt.show()
    f.close()


    #* Histogram Equalization

    sum_ = [0 for _ in range(256)]

    for i in range(256):
        for j in range(i+1):
            #print(hist[j])
            sum_[i] += hist[j] #* Accumulated value

    #print("sum : ")
    print(sum_)

    #* Normalization
    N = 512 * 512
    normalization = [0 for _ in range(256)]

    i=0
    for i in range(256):
        normalization[i] = int(sum_[i] * 255 / N)

    #print("normalization")
    print(normalization)


    f = open('lena_bmp_512x512_new.bmp', 'rb')

    header = f.read(0x430)
    data = f.read(1)
    newfile = open("newfile.bmp", "wb")

    newfile.write(header)

    while data != b"":
        data_after_eq = normalization[data[0]]
        
        if data_after_eq > 255:
            data_after_eq = 255
        if data_after_eq < 0:
            data_after_eq = 0
        byte = int.to_bytes(data_after_eq,1,'big')
        newfile.write(byte)
        
        data = f.read(1)

    newfile.close()
    f.close()

    x = range(len(normalization))
    plt.bar(x,normalization,width=1,color='skyblue',alpha=0.7,linewidth=4)

    plt.title('(b) Normalization', fontsize=12)

    plt.xlabel('Pixel Intersity', fontsize=10)

    plt.ylabel('frequency', fontsize=10)


    plt.show()

    f = open('newfile.bmp','rb')
    header = f.read(0x430)
    data = f.read(1)

    hist2 = [0 for _ in range(256)]
    while data != b"":

        hist2[data[0]] += 1
        data = f.read(1)

    x = range(len(hist2))
    plt.bar(x,hist2,width=1,color='green',alpha=0.7,linewidth=4)

    plt.title('(b) Equalization', fontsize=12)

    plt.xlabel('Pixel Intersity', fontsize=10)

    plt.ylabel('frequency', fontsize=10)

    plt.show()
    f.close()

    #* Basic Contrast Stretching

    #? get low value and high value
    low_strech = min_original
    high_stretch = max_original

    f = open('lena_bmp_512x512_new.bmp', 'rb')

    header = f.read(0x430)
    data = f.read(1)
    newfile = open("newfile_stretching.bmp", "wb")
    newfile.write(header)

    while data != b"":
        #print(data[0])
        new_pixel = int((data[0] - low_strech) * 255 / (high_stretch - low_strech))
        if new_pixel > 255:
            new_pixel = 255
        if new_pixel < 0:
            new_pixel = 0
        byte = int.to_bytes(new_pixel,1,'big')
        newfile.write(byte)
        data = f.read(1)


    f.close()
    newfile.close()

    hist_stretching = [0 for _ in range(256)]
    f = open('newfile_stretching.bmp', 'rb')

    header = f.read(0x430)
    data = f.read(1)
    

    while data != b"":
        
        hist_stretching[data[0]] += 1
        data = f.read(1)

    x = range(len(hist_stretching))
    plt.bar(x,hist_stretching,width=1,color='black',alpha=0.7,linewidth=4)

    plt.title('(c) Basic Contrast Stretching', fontsize=12)

    plt.xlabel('Pixel Intersity', fontsize=10)

    plt.ylabel('frequency', fontsize=10)

    plt.show()
    f.close()

    #* End-In-Search
    low_endin = 50
    high_endin = 190

    
    f = open('lena_bmp_512x512_new.bmp', 'rb')

    header = f.read(0x430)
    data = f.read(1)
    newfile = open("newfile_End_In_Search.bmp", "wb")
    newfile.write(header)

    while data != b"":
        #print(data[0])
        new_pixel = output(data[0],low=low_endin,high=high_endin)

        if new_pixel > 255:
            new_pixel = 255
        if new_pixel < 0:
            new_pixel = 0
        byte = int.to_bytes(new_pixel,1,'big')
        newfile.write(byte)
        data = f.read(1)


    f.close()
    newfile.close()

    hist_endinsearch = [0 for _ in range(256)]
    f = open('newfile_End_In_Search.bmp', 'rb')

    header = f.read(0x430)
    data = f.read(1)
    

    while data != b"":
        
        hist_endinsearch[data[0]] += 1
        data = f.read(1)

    x = range(len(hist_endinsearch))
    plt.bar(x,hist_endinsearch,width=1,color='orange',alpha=0.7,linewidth=4)

    plt.title('(d) End-In-Search', fontsize=12)

    plt.xlabel('Pixel Intersity', fontsize=10)

    plt.ylabel('frequency', fontsize=10)


    plt.show()
    f.close()


    print("############################################")
    print(hist_stretching)
    print("############################################")
    print(hist_endinsearch)

def output(x,low,high):
    #* for end-in search
    if x < low:
        result = 0
    elif x > high:
        result = 255
    else:
        result = (x-low) * 255 / (high-low)

    return int(result)

if __name__== '__main__':
    print("!")
    main()
