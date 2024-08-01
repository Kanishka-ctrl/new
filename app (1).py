# For tomato leaf disease prediction
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Render the custom CSS style

# Set page configuration and background color

st.set_page_config(
    page_title="TOMATO DISEASE PREDICTOR",
    page_icon=":tomato:",
    layout="centered",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
    <style>
    body {
        background-color: #000000;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

hide_streamlit_style = """
    <style>
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('tomatoes.h5')
    return model
with st.spinner('Model is being loaded..'):
    model=load_model()

st.write("""
         # TOMATO DISEASE PREDICTOR
         """
         )

file = st.file_uploader("", type=["jpg", "png"])
def import_and_predict(image_data, model):
        size = (256,256)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        img = np.asarray(image)
        img_reshape = img[np.newaxis,...]
        prediction = model.predict(img_reshape)
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    class_names = ['Early blight', 'Late blight', 'Healthy']
    string = "Prediction : " + class_names[np.argmax(predictions)]
    if class_names[np.argmax(predictions)] == 'Healthy':
        st.success(string)
    else:
        st.warning(string)


