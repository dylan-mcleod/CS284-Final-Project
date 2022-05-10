from osgeo import gdal 
import matplotlib.image as plimg
import matplotlib.pyplot as plt
import numpy as np


def SplitTIF(name, outfilename, iscover):
    dataset = gdal.Open(name, gdal.GA_ReadOnly) 
    band = dataset.GetRasterBand(1) 
    arr= band.ReadAsArray() 

    #####Class to RGB values for covertype images
    if iscover:
        ind_arr = np.array([[0.067,0.067,0.93], [0.0,0.334,0.0], [0.667,0.667,0.067], [0.2,0.667,0.2], [0.4,0.4,0.4]])
        RGBarr = ind_arr[arr][:]
        arr = RGBarr
    ######

    tiles = [arr[x:x+128,y:y+128] for x in range(0,arr.shape[0],128) for y in range(0,arr.shape[1],128)]
    print('Save start')
    ind = 0
    for i in range(len(tiles)-1):
        if len(tiles[i])==128 and len(tiles[i][0]) ==128:
            if not iscover:
                plimg.imsave(outfilename + "%s.png"%ind,tiles[i],vmin=0, vmax=8500,cmap="gray")
            else:
                plimg.imsave(outfilename + "%s.png"%ind,tiles[i])
            ind += 1
            if i%1000 == 0:
                print(i/1000)

SplitTIF('Data/DEMUganda.tif',   'Data/Uganda/elevation/Uganda', False)
SplitTIF('Data/CoverUganda.tif', 'Data/Uganda/cover/Uganda', True)




######RGB image to array for RGB images

#band2 = dataset.GetRasterBand(2)
#band3 = dataset.GetRasterBand(3)
#arr2 = band2.ReadAsArray()
#arr3 = band3.ReadAsArray()

#RGBarr = np.ndarray(shape=(arr.shape[0],arr.shape[1],3))
#RGBarr[:,:,2] = arr/10500  +0.05
#RGBarr[:,:,1] = arr2/10500 +0.05
#RGBarr[:,:,0] = arr3/10500 +0.0

#arr = RGBarr

######



##Cut up into 256x256 and save to designated filelocation

#img = plimg.imread('UgandaImages/elevation/uganda0.png')

# print(len(tiles))
#plt.imshow(img)

#plt.imshow(arr)
#plt.show()