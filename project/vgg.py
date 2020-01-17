from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as preprocess19
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as preprocess16
from keras.models import Model
import numpy as np
import os

def model_vgg19():
    base_model = VGG19(weights='imagenet')
    return Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)

def model_vgg16():
    return VGG16(weights='imagenet', include_top=False)

def model_vgg16_fully():
    return VGG16(weights='imagenet')

def model_vgg16_max():
    return VGG16(weights='imagenet', include_top=False, pooling='max')

def model_vgg16_avg():
    return VGG16(weights='imagenet', include_top=False, pooling='avg')

switcher={
        "VGG19":model_vgg19,
        "VGG16":model_vgg16,
        "VGG16_fully":model_vgg16_fully,
        "VGG16_max":model_vgg16_max,
        "VGG16_avg":model_vgg16_avg
    }

def getVGGModel(descriptor_type):
    return switcher[descriptor_type]()

def predict(model,img_path,descriptor_type):

    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    if descriptor_type == "VGG19" :
        x = preprocess19(x)
    else :
        x = preprocess16(x)

    return model.predict(x)
