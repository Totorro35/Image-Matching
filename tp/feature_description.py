import cv2
import numpy as np
from matplotlib import pyplot as plt



img = cv2.imread('../data/lena.jpg') # load a color image (default)
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# or
#gray = cv2.imread('../data/lena.jpg',0)



########### SIFT ###########


### Opencv 2.4.13
#sift = cv2.SIFT()

### Opencv 3
sift = cv2.xfeatures2d.SIFT_create()

kp, des = sift.detectAndCompute(gray,None)

img1 =cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

print("Number of SIFT descriptors : {0}".format(len(kp)))
print("SIFT dimension : {0}".format(len(des[0])))


########### SURF ###########

### Opencv 2.4.13
#surf = cv2.SURF(400)   # set Hessian Threshold to 400: the higher, the less keypoint detected
#surf.extended = False    # to get 64-dim descriptors else 128-dim descriptors

### Opencv 3
surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400, extended=False)

kp, des = surf.detectAndCompute(gray, None)

img2 =cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

print("Number of SURF descriptors : {0}".format(len(kp)))
print("SURF dimension : {0}".format(surf.descriptorSize()))


# display images with matplotlib
plt.subplot(1,2,1),plt.imshow(img1), plt.title('SIFT')
plt.subplot(1,2,2),plt.imshow(img2), plt.title('SURF')
plt.show()
