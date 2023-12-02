import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras.applications import ResNet50,ResNet152V2
import joblib
import numpy as np

model = ResNet152V2(weights='imagenet', include_top=False, pooling='avg')

model.save("relevance_checking_model.keras")

train_features_2d = np.load('train_features.npy')

# # Standardize features
scaler = StandardScaler()
train_features_std = scaler.fit_transform(train_features_2d)

np.save('train_features_std.npy', train_features_std)

scaler_filename = "scaler.save"
joblib.dump(scaler, scaler_filename) 




# And now to load...

# scaler = joblib.load(scaler_filename) 