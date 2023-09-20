import io
from PIL import Image
import streamlit as st
import pathlib
import cv2
from onnxruntime import InferenceSession
from PIL import Image
from opyv8 import Predictor
import pathlib
import numpy as np
from io import BytesIO    

model_path = pathlib.Path('custom_model/300best.onnx')
classes_path = model_path.parent.joinpath("model_classes.txt")


def load_image():
    uploaded_file = st.file_uploader(label=' ')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data,caption='Original Image', use_column_width=True)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


def load_model(model_path):
    classes = classes_path.read_text().split("\n")
    session = InferenceSession(model_path.as_posix(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    predictor = Predictor(session, classes)
    return predictor


def predict(predictor, image):
    
    # Perform prediction
    image.thumbnail((640, 640))

    prediction = predictor.predict(image)

    # Check the number of bounding boxes
    num_boxes = len(prediction.labels)
    print("Number of bounding boxes:", num_boxes)

    # Convert PIL image to OpenCV format (BGR)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Iterate over the bounding boxes and draw them on the image
    for bbox in prediction.labels:
        x, y, width, height = bbox.x, bbox.y, bbox.width, bbox.height
        class_name = bbox.classifier
        
        # Draw rectangle
        cv2.rectangle(image_cv, (x, y), (x + width, y + height), (0, 255, 0), 2)
        
        # Add class name as text on the bounding box
        cv2.putText(image_cv, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the image with bounding boxes using streamlit
    st.image(image_cv, channels="BGR",caption='Predicted Image', use_column_width=True)

def main():
    # Streamlit app code
    st.markdown("<h2 style='text-align: center;'>Model Demo</h2>", unsafe_allow_html=True)

    model = load_model(model_path)

    # Create two square containers using 'beta_columns' layout
    col1, col2 = st.columns(2)
    
    # Container 1: Display the original image
    with col1:
        st.markdown("<h3 style='text-align: center;'>Original Image</h3>", unsafe_allow_html=True)
        image = load_image()

    # Container 2: Display the predicted image
    with col2:
        st.markdown("<h3 style='text-align: center;'>Predicted Image</h3>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        result = st.button('Perform Prediction')
        
        if result:
            st.markdown("<br>", unsafe_allow_html=True)
            st.write('Calculating results...')
            prediction = predict(model, image)
if __name__ == '__main__':
    main()
