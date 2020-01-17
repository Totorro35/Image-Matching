import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../data/lena.jpg')
print(img.shape)

# averaging
blur = cv2.blur(img,(5,5))

# OpenCV represents RGB images as multi-dimensional NumPy arrays...but in reverse order! This means that images are actually represented in BGR order rather than RGB! Need to convert the image from BGR to RGB.
plt.subplot(121),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()

# gaussian blur
gblur = cv2.GaussianBlur(img,(5,5),0)

# median blur
median = cv2.medianBlur(img,5)

cv2.imwrite('blurred.png',blur)