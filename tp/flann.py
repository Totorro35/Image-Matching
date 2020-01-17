import cv2
import numpy as np
from timeit import default_timer as timer
import argparse

def main(dim,db_size,q_size,knn,FLANN_INDEX_ALGO):
    print("Dimension : {0}".format(dim))
    ####### Utilisation flann
    # - les matrices descripteurs ou requetes doivent etre du type 'np.float32'
    # - le vecteur requete doit etre correctement formate : qdesc.shape = (1, desc_dim)
    # -> reshape
    # distance (a,b) calculee par knnSearch = sum (ai-bi)^2
    ########

    #### Generer aleatoirement une matrice des descripteurs de la base
    # - randint(100) : distribution uniforme, nombres entiers entre 0 et 100
    # - size=(10,5) : 10 descripteurs de dimension 5
    mat_desc=np.array(np.random.randint(100,size=(db_size,dim)),dtype=np.float32)
    #print("Database descriptors : \n{0}".format(mat_desc))


    #### Generer un vecteur ou une matrice de vecteurs requete
    qdesc=np.array(np.random.randint(100,size=(q_size,dim)),dtype=np.float32)
    #print("Query descriptor : \n{0}".format(qdesc))

    ######## Database Descriptors indexing
    # FLANN parameters
    # Algorithms
    # 0 : FLANN_INDEX_LINEAR,
    # 1 : FLANN_INDEX_KDTREE,

    start = timer()
    index_params = dict(algorithm = FLANN_INDEX_ALGO)   # for linear search
    #index_params = dict(algorithm = FLANN_INDEX_ALGO, trees = 5) # for kdtree search

    ### OpenCV 3
    fl=cv2.flann_Index(mat_desc,index_params)

    end = timer()
    print("Indexing time: {0}".format(end - start))


    ######## Query search
    start = timer()
    #search_params = dict(checks=50)
    idx, dist=fl.knnSearch(qdesc,knn,params={})
    end = timer()
    print("Search time: {0}".format(end - start))


    #print idx.shape
    #print("indices \n{0}".format(idx))
    #print("distances \n{0}".format(dist))

    print("Distance to neareast neighbor = {0}".format(dist[0][0]))
    print("Distance to farthest neighbor = {0}".format(dist[0][db_size-1]))
    print("Ratio 1-nn and 2-nn = {0}".format(dist[0][0]/dist[0][1]))
    print("Ratio 1-nn and N-nn = {0}".format(dist[0][0]/dist[0][db_size-1]))

    #Thresholding the distance (Radius Search)
    #print idx[dist<3000]

    # 1. test the search with a set of query descriptors
    # 2. change the descriptors dimension (d=500) and the number of DB descriptors (N=1000)
    # 3. modify the DB size = 10 000, 100 0000, and 1 000 000 (change knn values accordingly)
    # 4. compare the computation time (indexing and search) in each case for both linear search and KDT tree

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dim", help="Dimension of Descriptor", type=int,default=2)
    parser.add_argument("-i", "--i_size", help="Nb desc by Img", type=int,default=1)
    parser.add_argument("-db", "--db_size", help="Size of DB", type=int,default=100000)
    parser.add_argument("-q", "--q_size", help="Nb of request", type=int,default=1)
    parser.add_argument("-k", "--knn", help="Nb of neighboors", type=int,default=100000)
    parser.add_argument("-a", "--algo", help="FLANN_INDEX_ALGO 0-linear;1-kdTree", type=str,default=0)

    args = parser.parse_args()

    return args

if __name__=="__main__":
    args = parseArguments()
    print("______________________________________")
    for i in (2,3,5,10,50,100,200,500,1000):
        main(i,args.i_size*args.db_size,args.i_size*args.q_size,args.knn,args.algo)
        print("______________________________________")