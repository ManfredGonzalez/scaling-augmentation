import cv2
import numpy as np
'''
This method is useful to split or slice an image into different pieces

sliceWidth, sliceHeight:
  These are the params that set the height and width of the slices in which the
  original image will be divided.

imagePath:
  This is the location path of the original image that will be split

imageName:
  This param is the seed name that the pieces will have, 
  for example: '00000000'

format:
  This will establish the file extension of the image for example .png or .jpg

destDir: 
  This is the path of the directory where the image slices will be saved
'''
def myImageSlicer(sliceWidth,sliceHeight,imagePath):
    y = 0
    x = 0
    width = sliceWidth
    height = sliceHeight
    ai1 = 0
    ai2 = 0
    img = cv2.imread(imagePath)
    originalHeight = img.shape[0]
    originalWidth = img.shape[1]
    images_matrix = np.empty(shape=(int(originalHeight/sliceHeight),int(originalWidth/sliceWidth))+(0,)).tolist()
    while y < originalWidth:
        while x < originalHeight:
            crop_img = img[x:x+height, y:y+width]
            images_matrix[ai2][ai1] = crop_img
            #cv2.imwrite('slice_0'+str(ai2)+'_0'+str(ai1)+'.jpg', crop_img)
            x = x + height
            ai2 = ai2 + 1
        
        x = 0
        y = y + width
        ai2 = 0
        ai1 = ai1 + 1

    return images_matrix