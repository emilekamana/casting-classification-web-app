# Import necessary libraries
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras.applications.resnet50 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from keras.models import load_model
import joblib

# Function for image preprocessing
# Define the image size for preprocessing
img_size = (300, 300)
print('running')

def extract_features(image_path):
    image =tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)

    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    # Preprocess the images for ResNet50
    preprocessed_images = preprocess_input(img_array)
    
    relevance_model = load_model("relevance_checking_model.keras")
    print('uploaded model')

    # Extract features using ResNet50
    features = relevance_model.predict(preprocessed_images)

    return features

# Function to check if an image is relevant based on clustering
def is_relevant_by_cosine_similarity(image_path):
    test_features=extract_features(image_path)
    
    scaler = joblib.load("scaler.save") 
    print('uploaded scaler')

    # Standardize features
    image_features_std = scaler.transform(test_features)

    
    train_features_std = np.load('train_features_std.npy')
    print('uploaded features')

    similarities = cosine_similarity(image_features_std, train_features_std)
    max_similarity = np.max(similarities)
    average_similarity = np.mean(similarities)
    min_similarity = np.min(similarities)
    median_similarity = np.median(similarities)

    print("average similarity: ", average_similarity)
    print("max similarity: ", max_similarity)
    print("min similarity: ", min_similarity)
    print("median similarity: ", median_similarity)

    # You can adjust the threshold as needed based on your dataset
    if max_similarity >= 0.80:
        return True
    else:
        return False

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

    if st.button("Predict"):
        # Check if the uploaded file is an image
        if uploaded_file is not None:
            try:

                if is_relevant_by_cosine_similarity(uploaded_file):
                # Add a button to check the uploaded file
                    # Display the uploaded image
                    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

                    # Preprocess the image
                    processed_image = preprocess_image(uploaded_file)

                    # Load the saved model
                    model = load_model("best_fine_tuned_model.h5")

                    # Make predictions
                    prediction = predict_defect(processed_image, model)

                    # Display prediction result
                    st.success("Prediction: " + prediction)
                else:
                    st.error("The file you uploaded is not relevant. Please upload a valid image file.")

            except:
                # If the file is not an image, show an error message
                st.error("Something went wrong. Please try uploading another image file.")
        else:
            # If no file is uploaded, show a warning message
            st.warning("Please upload an image file to make a prediction.")


if __name__ == '__main__':
    main()
