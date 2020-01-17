#export PYTHONPATH="/usr/local/opencv-2.4.9/lib/python2.7/site-packages/"
import os
import sys
import glob
import cv2
import numpy as np
import argparse

import vgg

from timeit import default_timer as timer

def parseArgument():
    parser = argparse.ArgumentParser()

    ## Database name
    parser.add_argument("-d", "--database", dest="db_name",
                        help="input image database", metavar="STRING", default="None")
    parser.add_argument("-s", "--save", dest="save",
                        help="Output database", metavar="STRING", default="None")
    parser.add_argument("-t", "--descriptor", dest="descriptor",
                        help="Descriptor method", metavar="STRING", default="None")

    return parser.parse_args()

args = parseArgument()

## Set paths
img_dir= args.db_name + "/"
imagesNameList = glob.glob(img_dir+"*.jpg")
output_dir=args.save

if not os.path.exists(img_dir):
    msg="The directory containing images: "+img_dir+" is not found -- EXIT\n"
    print(msg)
    sys.exit(1)

if not os.path.exists(output_dir):
    print("The output directory does not exist")
    os.mkdir(output_dir)
    print("Directory created")

print("============================\n")
print("     DB Indexing\n")
print("============================")

db_desc=[]
db_name=[]

total_time=0

print("Feature Extraction :")
if args.descriptor == "SIFT":
    total_kp=0
    sift = cv2.xfeatures2d.SIFT_create()
    for imgName in imagesNameList:
        img = cv2.imread(imgName)
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
        start = timer()
        kp, des = sift.detectAndCompute(gray,None)
        end = timer()
        total_time+=end-start
        if not des is None:
            for desc in des :
                db_desc.append(desc)
                db_name.append(imgName)
        total_kp+=len(kp)
        print("     {0} -> {1} descriptors".format(imgName,len(kp)))

    
    print(" Total descriptors : {0}".format(total_kp))
    print(" Mean descriptors : {0}".format(total_kp/len(imagesNameList)))

else :
    model = vgg.getVGGModel(args.descriptor)
    for imgName in imagesNameList:
        start = timer()
        desc = vgg.predict(model,imgName,args.descriptor)
        end = timer()
        total_time+=end-start
        db_desc.append(desc)
        db_name.append(imgName)
        print("     {0} -> {1} descriptors".format(imgName,desc.shape))

print(" Total time : {0} s".format(total_time))

np.save(output_dir+"db_name.npy",db_name)
np.save(output_dir+"db_desc.npy",db_desc)

print("Indexing :")

FLANN_INDEX_ALGO=1
index_params = dict(algorithm = FLANN_INDEX_ALGO)   # for linear search

start = timer()
fl=cv2.flann.Index(np.asarray(db_desc,dtype=np.float32),index_params)
end = timer()
print(" Total time : {0} s".format(end-start))
fl.save(output_dir+"db_index.dat")








