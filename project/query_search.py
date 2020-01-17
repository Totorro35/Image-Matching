import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import sys
from timeit import default_timer as timer
import argparse

def getImageId(imname):
    DB_TYPE="COREL"
    if (DB_TYPE == "COREL"):
        Id = imname.split('_')[1]
    elif (DB_TYPE == "NISTER"):
        Id = imname.split('-')[1]
    elif (DB_TYPE == "Copydays"):
        Id = imname.split('_')[-2]
    else:
        Id = imname.split('.')[-1]
    
    return Id

def indexToImage(index,db_name):
    i=0;
    last=""
    for (sum_idx, path) in db_name:
        if int(sum_idx)>index:
            return (i,last)
        last=path
        i=i+1

def parseArgument():
    parser = argparse.ArgumentParser()

    ## Database name
    parser.add_argument("-d", "--database", dest="db",
                        help="input descriptor database", metavar="STRING", default="None")
    parser.add_argument("-q", "--query", dest="query",
                        help="Query image", metavar="STRING", default="None")
    parser.add_argument("-t", "--test", dest="test",
                        help="Test the indexing (Precision, Recall, ...)", action="store_true")

    return parser.parse_args()

args = parseArgument()

db_desc=np.load(args.db+"db_desc.npy")
db_name=np.load(args.db+"db_name.npy")

fl = cv2.flann.Index()
fl.load(db_desc,args.db+"db_index.dat")

queryImage = cv2.imread(args.query)

plt.figure(0), plt.title("Image requete")
plt.imshow(cv2.cvtColor(queryImage, cv2.COLOR_BGR2RGB))
plt.show()

sift = cv2.xfeatures2d.SIFT_create()
gray= cv2.cvtColor(queryImage,cv2.COLOR_BGR2GRAY)
kp, qdesc = sift.detectAndCompute(gray,None)

KNN=5
idx, dist=fl.knnSearch(qdesc,KNN,params={})

result={}

for i in range(len(idx)):
    for j in range(KNN):
        path_image = db_name[idx[i][j]]
        score = dist[i][j]
        if path_image in result:
            result[path_image]+=1
        else :
            result[path_image]=1

filtered_scores = sorted(result.items(), key=lambda item: item[1], reverse=True)

###########################
#### Display the top images
###########################
top=10
plt.figure(1), plt.title("SIFT")
for i in range(top):
    img = cv2.imread(filtered_scores[i][0])
    score = filtered_scores[i][1]
    plt.subplot(2,5,i+1),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('rank '+str(i+1)), plt.xticks([]), plt.yticks([]),plt.xlabel(str(score))

#plt.savefig(resfilename + "_top" + str(top) +".png")
plt.show()

if(not args.test):
    exit()

queryId=getImageId(args.query)
print("Identifiant de la requete : {0}".format(queryId))

precision = np.zeros(len(filtered_scores), dtype=float)
recall = np.zeros(len(filtered_scores), dtype=float)

nbMaxRelevantImage=7
nbActualRelevantImage=0
cpt=0

for (path_name,score) in filtered_scores:
    cpt+=1
    if(getImageId(path_name) == queryId):
        nbActualRelevantImage+=1
    precision[cpt-1]=nbActualRelevantImage/cpt
    recall[cpt-1]=nbActualRelevantImage/nbMaxRelevantImage

plt.clf()
plt.plot(recall, precision, lw=2, color='navy',
         label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.05])
plt.title('Precision-Recall for SIFT')
plt.legend(loc="upper right")
#plt.savefig(resfilename + "_rp.png")
#plt.savefig(output_dir + args.query_name + "_rp.pdf", format='pdf')
plt.show()
