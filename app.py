import streamlit as st
import os
from matplotlib import image

def main():
    # Set page title and introduction
    st.set_page_config(page_title='Doctor Green', page_icon='🌱')

    # Set page title and introduction
    st.title("Doctor Green")

    # Absolute path to the directory containing this script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Absolute path to the 'resources' directory
    RESOURCES_DIR = os.path.join(BASE_DIR, "resources")
    # Absolute path to the 'images' directory
    IMAGES_DIR = os.path.join(RESOURCES_DIR, "images")
    # Absolute path to the image file
    IMAGE_PATH = os.path.join(IMAGES_DIR, "plant.jpeg")

    img = image.imread(IMAGE_PATH)
    st.image(img)

    st.markdown(
        """
        <style>
        h1 {
        color: #00FF00;
        }
        .title {
            font-family: 'Arial Black', sans-serif;
            font-size: 72px;
            color: #ffffff;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
            text-align: center;
            margin-top: 100px;
        }
        body {
            background-color: #2ecc71;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()
