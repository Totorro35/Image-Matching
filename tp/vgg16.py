from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os

model = VGG16(weights='imagenet')

img_path = '../data/lena.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)
print("last fully connected size : {0}".format(features.shape))

model = VGG16(weights='imagenet', include_top=False)
features = model.predict(x)
print("last convolutionnal size : {0}".format(features.shape))

model = VGG16(weights='imagenet', include_top=False, pooling='avg')
features = model.predict(x)
print("last convolutionnal average pooling size : {0}".format(features.shape))

model = VGG16(weights='imagenet', include_top=False, pooling='max')
features = model.predict(x)
print("last convolutionnal max pooling size : {0}".format(features.shape))

output_dir = "."
name, _ = os.path.splitext(img_path)
feat_filepath = os.path.join(output_dir, name+'.npy')

with open(feat_filepath, 'wb') as f:
    np.save(f, features)