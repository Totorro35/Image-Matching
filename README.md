# VO Ewa

## Article

Etude d'un [article scientifique](ARTICLE.md)

[15] Learning to Generate Chairs with Convolutional Neural Networks. Alexey Dosovitskiy, J. Springenberg, Thomas Brox. CVPR 2015.

## Projet

### Installation
`export DOCKER_NAME = name`  
`docker build -t ${DOCKER_NAME}:v1 .`  
`docker run -v $PWD:/workspace/VO_EWA -w /workspace/VO_EWA -it ${DOCKER_NAME}:v1 /bin/bash`  

### Indexing
`python3 project/db_indexing.py --database data/base1/ --save Base/Base1/ --descriptor "SIFT"`  

Choix de descripteur disponible :
- SIFT
- VGG19
- VGG16
- VGG16_fully
- VGG16_max
- VGG16_avg

### Query
Fonctionne uniquement pour les descripteurs SIFT  
`python3 project/query_search.py --database Base/COREL/ --query data/COREL/corel_0000000303_512.jpg --test`  
`--test` option pour calculer la courbe precision-rappel