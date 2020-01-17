import numpy as np
import cv2
from matplotlib import pyplot as plt
import argparse

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i1", "--img1", help="input image 1", type=str)
    parser.add_argument("-i2", "--img2", help="input image 2", type=str)
    parser.add_argument("-m", "--method", help="method augmentation", type=str)

    args = parser.parse_args()

    return args

def main(img1,img2):

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT within the first image
    kp1, des1 = sift.detectAndCompute(img1,None)
    txt="In img1 : "+ str(len(des1))+ "descriptors \n"
    print(txt)
    img1kp=cv2.drawKeypoints(img1,kp1,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure(1)
    plt.imshow(img1kp),plt.title('Keypoints of Image 1')


    # Do the same for the second image
    kp2, des2 = sift.detectAndCompute(img2,None)
    txt="In img2 : "+ str(len(des2))+ "descriptors \n"
    print(txt)
    img2kp=cv2.drawKeypoints(img2,kp2,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure(2)
    plt.imshow(img2kp),plt.title('Keypoints of Image 2')

    # Search the k-nn of each descriptor if img1
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    txt="Number of matches : "+ str(len(matches))
    print(txt)

    # For the fun you can simply try this:
    # matches = bf.knnMatch(des2, des1, k=2)

    # Nicely presenting the results:
    #
    # Need to draw only good matches, so create a mask
    # non matches are displayed but not linked.
    matchesMask = [[0,0] for i in range(len(matches))]

    nbMatches = 0
    # Selection of good matches - TODO
    for i,(m,n) in enumerate(matches):
        #m.distance = distance of the query desc queryIdx to its 1-nn (trainIdx)
        #n.distance = distance of the query desc queryIdx to its 2-nn (trainIdx)
        if m.distance <= 0.75 * n.distance :
            matchesMask[i]=[1,0]
            nbMatches+=1

    draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    txt="There are "+ str(nbMatches)+ " matches according to the matching criteria\n"
    print(txt)
    plt.figure(3)
    plt.imshow(img3),plt.title('Matches')
    plt.show()

    # 1. Compare different matching criteria
    # 2. Compare number of matches between similar images and dissimilar images
    # 3. Try with different images: textured, logos, faces...
    # 4. Using Gimp and/or opencv, produce attacked versions of an image: blur, crop, rotation, jpeg compression with various quality factors, occultations, flip, noise
    # 5. Check which transformations are the most difficult for correctly matching descriptors

def augmentation(method,img):
    if method == "blur":
        return cv2.blur(img,(5,5))
    elif method == "rotate":
        return cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
    elif method == "flip":
        return cv2.flip(img,0)
    elif method == "noise":
        gaussian_noise = img.copy()
        cv2.randn(gaussian_noise, 0, 50)
        return img+gaussian_noise
    elif method == "inverse":
        return 255-img
    elif method == "BGR":
        split_src = cv2.split(img)
        if(len(split_src)==3):
            return cv2.merge((split_src[2], split_src[1], split_src[0]))
        else:
            return img
    else :
        print("No method")
        return img

if __name__=="__main__":
    args = parseArguments()
    img1 = cv2.imread(args.img1,0)
    img2 = cv2.imread(args.img2,0)
    img1=augmentation(args.method,img1)
    main(img1,img2)