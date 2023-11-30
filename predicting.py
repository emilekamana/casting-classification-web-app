# Import necessary libraries
import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model

# Function for image preprocessing
# Define the image size for preprocessing
img_size = (300, 300)


def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize(img_size)  # Resize image to the specified dimensions
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


# Function for making predictions
def predict_defect(image_array, model):
    prediction = model.predict(image_array)
    if prediction >= 0.5:
        predicted_label = "defective"
        return "The image is " + predicted_label

    # Predicted Class : OK
    else:
        predicted_label = "ok"
        return "The image is " + predicted_label


# Main Streamlit app
def main():
    st.title("Defect Casting Prediction app")

    # Upload image through Streamlit
    # Allow jpg, png, and jpeg file types
    uploaded_file = st.file_uploader("Upload an image of type jpg, png, or jpeg...", type=["jpg", "png", "jpeg"])

    # Add a button to check the uploaded file
    if st.button("Predict"):
        # Check if the uploaded file is an image
        if uploaded_file is not None:
            try:
                # Display the uploaded image
                st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

                # Preprocess the image
                processed_image = preprocess_image(uploaded_file)

                # Load the saved model
                model = load_model("best_fine_tuned_model.h5")

                # Make predictions
                prediction = predict_defect(processed_image, model)

                # Display prediction result
                st.success("Prediction Probability:", prediction)

            except:
                # If the file is not an image, show an error message
                st.error("The file you uploaded is not a valid type. Please upload a valid image file.")
        else:
            # If no file is uploaded, show a warning message
            st.warning("Please upload an image file to make a prediction.")


if __name__ == '__main__':
    main()
