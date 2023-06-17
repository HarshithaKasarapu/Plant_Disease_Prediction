import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
from PIL import Image

#from tensorflow.keras.optimizers import legacy

# Set the path to the directories
train_data_dir = 'E:/harshitha/plant_project/Plant 2/Plant/model/op1/train'
valid_data_dir = 'E:/harshitha/plant_project/Plant 2/Plant/model/op1/valid'
img_width, img_height = 224, 224
num_classes = 38
batch_size = 32
epochs = 10

# Load the pre-trained VGG16 model without the top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top of the base model
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=output) 

#optimizer = legacy.Adam()
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load the model weights
model.load_weights('E:/harshitha/plant_project/Plant 2/Plant/model/plant_model')
labels = ['Apple_Apple_scab','Apple_Black_rot','Apple_Cedar_apple_rust','Apple_healthy','Blueberry_healthy',
'Cherry_(including_sour)_Powdery_mildew','Cherry_(including_sour)_healthy','Corn_(maize)_Cercospora_Gray_leaf_spot',
'Corn_(maize)Common_rust','Corn_(maize)_Northern_Leaf_Blight','Corn_(maize)_healthy','Grape_Black_rot','Grape_Esca_(Black_Measles)',
'Grape_Leaf_blight_(Isariopsis_Leaf_Spot)','Grape_healthy','Orange_Haunglongbing_(Citrus_greening)','Peach_Bacterial_spot',
'Peach_healthy','Pepper_bell_Bacterial_spot','Pepper,_bell_healthy','Potato_Early_blight','Potato_Late_blight','Potato_healthy',
'Raspberry_healthy','Soybean_healthy','Squash_Powdery_mildew','Strawberry_Leaf_scorch','Strawberry_healthy',
'Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_Late_blight','Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot',
'Tomato_spider_mites Two-spotted_spider_mite','Tomato_Target_Spot','Tomato_Tomato_Yellow_Leaf_Curl_Virus','Tomato_Tomato_mosaic_virus','Tomato_healthy']


st.markdown("<h1 style='color: #7E24F7;'>PLANT CLASSIFICATION</h1>", unsafe_allow_html=True)

st.set_option('deprecation.showfileUploaderEncoding', False)


# Create the image data generator for prediction
test_datagen = ImageDataGenerator(rescale=1. / 255)

#class_info
class_info_df = pd.read_csv('E:/harshitha/plant_project/Plant 2/Plant/resources/data/class_info_2.csv')    
class_info = class_info_df.set_index('predicted_class').to_dict(orient='index')

# Streamlit app

st.sidebar.markdown("<h2 style='color: #9776FF; font-weight: bold;'>Pick your Choice</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<h2 style='color: #DED3FF;'>Upload an image for prediction:</h2>", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

def main():
    if uploaded_file is not None:
    # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        # Preprocess the uploaded image for prediction
        img = Image.open(uploaded_file)
        img = img.resize((img_width, img_height))
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

    # Make a prediction using the loaded model
        prediction = model.predict(img_array)
        predicted_class = labels[tf.argmax(prediction, axis=1)[0]]

        # Display the predicted class

        predicted_class_text = f"<span style='font-size: 30px; color: #9212F6; font-weight: bold;'>Predicted Class:</span><br><span style='font-size: 24px; font-weight: bold;'>{predicted_class}</span>"
        st.markdown(predicted_class_text, unsafe_allow_html=True)

        style = "font-size: 24px; color: #CD12F6; font-weight: bold;"

        st.markdown(f"<span style='{style}'><b>Description:</b></span>", unsafe_allow_html=True)
        st.write(class_info[predicted_class]['Description'])

        symptoms = class_info[predicted_class]['Symptoms']
        symptoms_list = symptoms.split(".")
        symptoms_list = [symptom.strip() for symptom in symptoms_list if symptom.strip()]

        symptoms_text = "<span style='font-size: 24px; color: #CD12F6; font-weight: bold;'><b>Symptoms:</b></span><ul>"
        for symptom in symptoms_list:
            symptoms_text += f"<li>{symptom}</li>"
        symptoms_text += "</ul>"

        st.markdown(symptoms_text, unsafe_allow_html=True)        

        st.markdown(f"<span style='{style}'><b>Causes:</b></span>", unsafe_allow_html=True)
        st.write(class_info[predicted_class]['Causes'])


if __name__ == '__main__':
    main()
