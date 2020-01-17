#export PYTHONPATH="/usr/local/lib/python2.7/site-packages"

import numpy as np
import cv2
from matplotlib import pyplot as plt


############ UTILS
# Load an image in grayscale
#cv2.IMREAD_COLOR : Loads a color image. Any transparency of image will be neglected. It is the default flag.
#cv2.IMREAD_GRAYSCALE : Loads image in grayscale mode
# Instead of these three flags, you can simply pass integers 1, 0
img = cv2.imread('../data/lena.jpg',0)

# print image size
print(img.shape)

# display an image with opencv (until key pressed)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()



############  Compute grayscale histogram
# cv2.calcHist(image, channels, mask, histSize, ranges[])
# image : the source image. It should be given in square brackets, ie, "[img]".
# channels : the index of channel for which we calculate histogram. For example, if input is grayscale image, its value is [0]. For color image, you can pass [0],[1] or [2] to calculate histogram of blue,green or red channel respectively. It is also given in square brackets.
# mask : mask image. To find histogram of full image, it is given as "None". But if you want to find histogram of particular region of image, you have to create a mask image for that and give it as mask.
# histSize : this represents the BIN count. Need to be given in square brackets. For full scale, we pass [256].
# ranges : this is the data RANGE. Normally, it is [0,255].
#

hist = cv2.calcHist([img],[0],None,[256],[0,256])

# L2 normalization
cv2.normalize(hist,hist,norm_type=cv2.NORM_L2)

plt.plot(hist,color = 'b')
plt.title('Grayscale histogram'),plt.xlim([0,256])
plt.show()


############ Distances between histograms
img2 = cv2.imread('../data/box.png',0)
hist2 = cv2.calcHist([img2],[0],None,[256],[0,256])
cv2.normalize(hist2, hist2, norm_type=cv2.NORM_L2)

plt.plot(hist2,color = 'r')
plt.xlim([0,256])
plt.show()

dist=cv2.norm(hist, hist2, normType=cv2.NORM_L2, mask=None)
print("Distance between h1 and h2 = %.9f" % dist)

# 1. Compute the distance between 'lena' and 'messigray'



############  Color histogram
colimg = cv2.imread('../data/lena.jpg',1)
colhist = cv2.calcHist([colimg], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
print("3D histogram shape: {0}, with {1} values".format(colhist.shape, colhist.flatten().shape[0]))

cv2.normalize(colhist,colhist, norm_type=cv2.NORM_L2)
plt.plot(colhist.flatten(),color = 'b')
plt.title('Color histogram'),plt.xlim([0,colhist.flatten().shape[0]])
plt.show()

# 2. Plot the color histograms of 'lena' and 'baboon' on the same figure
# 3. Compute the distance between color histograms of 'lena' and 'baboon'
# 4. Compare with the distance between grayscale histograms of 'lena' and 'baboon'
