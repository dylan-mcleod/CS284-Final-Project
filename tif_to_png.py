from osgeo import gdal 
import matplotlib.image as plimg
import matplotlib.pyplot as plt
import numpy as np

dataset = gdal.Open('SierraDEM.tif', gdal.GA_ReadOnly) 
band = dataset.GetRasterBand(1) 
arr= band.ReadAsArray() 



######RGB image to array for RGB images


# band2 = dataset.GetRasterBand(2)
# band3 = dataset.GetRasterBand(3)
# arr2 = band2.ReadAsArray()
# arr3 = band3.ReadAsArray()


# RGBarr = np.ndarray(shape=(arr.shape[0],arr.shape[1],3))
# RGBarr[:,:,2] = arr/10500 +0.05
# RGBarr[:,:,1] = arr2/10500 +0.05
# RGBarr[:,:,0] = arr3/10500 +0.05

#arr = RGBarr

######



#####Class to RGB values for covertype images
##Very inefective, but works.

# RGBarr = np.ndarray(shape=(arr.shape[0],arr.shape[1],3))
# for x in range(arr.shape[0]):
#     for y in range(arr.shape[1]):
#         if arr[x,y] == 0:
#             RGBarr[x,y] = [0.067,0.067,0.93]
#         if arr[x,y] == 1:
#             RGBarr[x,y] = [0.0,0.334,0.0]
#         if arr[x,y] == 2:
#             RGBarr[x,y] = [0.667,0.667,0.067]
#         if arr[x,y] == 3:
#             RGBarr[x,y] = [0.2,0.667,0.2]
#         if arr[x,y] == 4:
#             RGBarr[x,y] = [0.4,0.4,0.4]
# arr = RGBarr

######
 

##Cut up into 256x256 and save to designated filelocation

tiles = [arr[x:x+256,y:y+256] for x in range(0,arr.shape[0],256) for y in range(0,arr.shape[1],256)]
print('Save start')
ind = 0
for i in range(len(tiles)-1):
    if len(tiles[i])==256 and len(tiles[i][0]) ==256:
        plimg.imsave("SierraImages/elevation/sierra%s.png"%ind,tiles[i])#,vmin=0, vmax=8500,cmap="gray") ##Uncomment when saviing hight map.
        ind += 1
        if i%1000 == 0:
            print(i/1000)


#img = plimg.imread('UgandaImages/elevation/uganda0.png')

# print(len(tiles))
#plt.imshow(img)
plt.imshow(arr)
plt.show()




