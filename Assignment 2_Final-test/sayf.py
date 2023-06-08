import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score

# Rotate Data Function
def rotate_data(data, angle):
    rotated_data = np.concatenate((data[:, angle:], data[:, :angle]), axis=1)
    return rotated_data

# Add Noise Function
def add_noise(data, mean, std_dev):
    noisy_data = data + np.random.normal(mean, std_dev, size=data.shape)
    return noisy_data

# Read the data
df_features = pd.read_csv('traindata.txt', delimiter=',', header=None)
df_labels = pd.read_csv('testlabels.txt', header=None)

# Convert the data to numpy arrays
X_test = df_features.to_numpy()
y_test = np.ravel(df_labels)

# Rotate the test data
X_test = rotate_data(X_test, angle=10)

# Add noise to the test data
X_test = add_noise(X_test, 0, 0.1)

# Define some constants
INPUT_SHAPE = (71,)  # Number of input features
NUM_CLASSES = 10  # Number of output classes (0-9)

# Load the saved model
model = tf.keras.models.load_model('model.h5')

# Make predictions on the test set
y_test_pred_prob = model.predict(X_test)
y_test_pred = np.argmax(y_test_pred_prob, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_test_pred)
print('Model accuracy: {:.2%}'.format(accuracy))
