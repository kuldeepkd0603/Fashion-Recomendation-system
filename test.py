import pickle
import numpy as np
import tensorflow
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import cv2

feature_list=np.array(pickle.load(open("embadding.pkl","rb")))
filenames=pickle.load(open("filenames.pkl",'rb'))
#print(feature_list.shape)

model=ResNet50(weights="imagenet",include_top=False,input_shape=(224,224,3))
model.trainable=False


model=tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])




img=image.load_img("sample/belt.webp",target_size=(224,224))
img_array=image.img_to_array(img)
expanded_img_array=np.expand_dims(img_array,axis=0)
preprocessed_img=preprocess_input(expanded_img_array)
result=model.predict(preprocessed_img).flatten()
normalized_result=result/norm(result)


neighbors=NearestNeighbors(n_neighbors=5,algorithm='brute',metric='euclidean')
neighbors.fit(feature_list)


distances,indices=neighbors.kneighbors([normalized_result])

for  file in indices[0]:
    temp_img=cv2.imread(filenames[file])
    cv2.imshow('output',cv2.resize(temp_img,(512,512)))
    
    cv2.waitKey(0)


print(indices)