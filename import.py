import cv2 
import glob

path = 'imgs/*.png'

imL = glob.glob(path)

for im in imL:
    image = cv2.imread(im)
    image = cv2.resize(image,(968,968),interpolation = cv2.INTER_AREA)
    #image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    cv2.imwrite(im, image)
