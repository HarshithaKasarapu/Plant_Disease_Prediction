import streamlit as st
import os
from matplotlib import image
import pandas as pd
import numpy as np
import plotly.express as px
import re
import plotly.express as px


# absolute path to this file
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# absolute path to this file's root directory
PARENT_DIR = os.path.join(FILE_DIR, os.pardir)
# absolute path of directory_of_interest
dir_of_interest = os.path.join(PARENT_DIR, "resources")

IMAGE_PATH = os.path.join(dir_of_interest, "images", "plant.jpeg")

img = image.imread(IMAGE_PATH)
st.image(img)

def main():
    # Set page title and introduction
    st.title("Plant Disease Classification")
    st.markdown(
        """
        <style>
        .title {
            color: #1f618d;
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        .intro {
            color: #34495e;
            font-size: 18px;
            margin-bottom: 2rem;
        }
        .section-header {
            color: #2c3e50;
            font-size: 24px;
            font-weight: bold;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        .section-content {
            color: #34495e;
            font-size: 16px;
            margin-bottom: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Display the introduction section
    st.markdown(
        """
        <div class="intro">
        This application performs plant disease classification using machine learning.
        It aims to identify various diseases that affect plants based on input images.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Display the project overview section
    st.markdown('<div class="section-header">Project Overview</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-content">
        Plant disease classification plays a vital role in agriculture by helping farmers identify and manage diseases in their crops.
        This application leverages machine learning techniques to automate the disease identification process, providing quick and accurate results.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Display the benefits section
    st.markdown('<div class="section-header">Why Plant Disease Classification?</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-content">
        Plant disease classification offers several benefits:
        <ul>
        <li>Early detection of diseases allows for timely intervention and prevention.</li>
        <li>Accurate disease identification helps farmers choose appropriate treatment methods.</li>
        <li>Increased crop yield and quality through effective disease management.</li>
        <li>Efficient use of resources by targeting specific diseases for treatment.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Display the application features section
    st.markdown('<div class="section-header">Application Features</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-content">
        The plant disease classification application provides the following features:
        <ul>
        <li>Upload an image of a plant sample to be classified.</li>
        <li>Preprocess the image to enhance features and remove noise.</li>
        <li>Apply a trained machine learning model to classify the plant sample.</li>
        <li>Display the predicted disease class and provide relevant information about the disease.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
    
if __name__ == '__main__':
    main()