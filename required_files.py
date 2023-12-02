import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras.applications import ResNet50,ResNet152V2
import joblib
import numpy as np

model = ResNet152V2(weights='imagenet', include_top=False, pooling='avg')

model.save("relevance_checking_model.h5")



# And now to load...

# scaler = joblib.load(scaler_filename) 