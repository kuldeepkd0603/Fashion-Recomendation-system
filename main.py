import streamlit as st
import os
import json
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm


feature_list = np.array(pickle.load(open('embadding.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))


model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([model, GlobalMaxPooling2D()])

st.title('Fashion Recommender System')


def save_uploaded_file(uploaded_file):
    try:
        os.makedirs('uploads', exist_ok=True)
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except Exception as e:
        st.error(f"Error saving uploaded file: {e}")
        return 0


def feature_extraction(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)
        return normalized_result
    except Exception as e:
        st.error(f"Error extracting features from image {img_path}: {e}")
        return None


def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices


def load_filtered_product_details(image_path, required_fields):
    json_path = os.path.join('styles', os.path.splitext(os.path.basename(image_path))[0] + '.json')
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                
                important_data = {
                    "id": data.get("data", {}).get("id"),
                    "price": data.get("data", {}).get("price"),
                    "discountedPrice": data.get("data", {}).get("discountedPrice"),
                    "productDisplayName": data.get("data", {}).get("productDisplayName"),
                    "brandName": data.get("data", {}).get("brandName"),
                    "baseColour": data.get("data", {}).get("baseColour"),
                    "fashionType": data.get("data", {}).get("fashionType"),
                    "season": data.get("data", {}).get("season"),
                    "year": data.get("data", {}).get("year"),
                    "usage": data.get("data", {}).get("usage"),
                    "landingPageUrl": data.get("data", {}).get("landingPageUrl"),
                    "defaultImageURL": data.get("data", {}).get("styleImages", {}).get("default", {}).get("imageURL")
                }
                return important_data
        except Exception as e:
            st.error(f"Error reading JSON file: {e}")
    else:
        st.error(f"JSON file not found: {json_path}")
    return None


uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
       
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption='Uploaded Image', use_column_width=True)
        
        
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
        if features is not None:
            st.text(f"Extracted features: {features}")
            
            
            indices = recommend(features, feature_list)
            st.text(f"Recommended indices: {indices}")
            
            
            if 'product_index' not in st.session_state:
                st.session_state['product_index'] = 0

            if st.button("Next Product"):
                st.session_state['product_index'] = (st.session_state['product_index'] + 1) % len(indices[0])

            current_index = st.session_state['product_index']
            image_path = filenames[indices[0][current_index]]
            if os.path.exists(image_path):
                #st.image(image_path, caption='Recommended Product', use_column_width=False, width=300)
                
               
                required_fields = [
                    'id', 'price', 'discountedPrice', 'productDisplayName', 'brandName',
                    'baseColour', 'fashionType', 'season', 'year', 'usage', 'landingPageUrl', 'defaultImageURL'
                ]
                
                
                product_details = load_filtered_product_details(image_path, required_fields)
                if product_details:
                    st.image(product_details.get('defaultImageURL', 'N/A'), caption='Recommended Product', use_column_width=False, width=300)
                    for field in required_fields:
                        
                        st.write(f"**{field.replace('_', ' ').title()}**: {product_details.get(field, 'N/A')}")
                else:
                    st.write("No details available for this product.")
            else:
                st.error(f"Image file not found: {image_path}")
    else:
        st.header("Some error occurred in file upload")
