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

### Query