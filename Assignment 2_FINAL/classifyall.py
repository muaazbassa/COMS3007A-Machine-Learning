import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import tensorflow as tf


def write_predictions_to_file(X_test, ensemble, filename):
    y_test_pred_prob = np.zeros((len(X_test), NUM_CLASSES))
    for model in ensemble:
        y_test_pred_prob += model.predict(X_test)

    y_test_pred = np.argmax(y_test_pred_prob, axis=1)

    with open(filename, 'w') as f:
        for label in y_test_pred:
            f.write(str(label) + '\n')

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

# Convert the data to numpy arrays
X_test = df_features.to_numpy()

# Data augmentation - random perturbations
augmented_X_test = []

for i in range(len(X_test)):
    original_data = X_test[i]
    augmented_X_test.append(original_data)

    # Apply random perturbations
    perturbed_data = original_data + np.random.normal(0, 0.1, size=original_data.shape)
    augmented_X_test.append(perturbed_data)

# Convert augmented data to numpy arrays
augmented_X_test = np.array(augmented_X_test)

# Concatenate augmented data with original data
X_test = np.concatenate([X_test, augmented_X_test], axis=0)

# Shuffle the augmented data
X_test = shuffle(X_test, random_state=42)

# Rotate the data
X_test = rotate_data(X_test, angle=10)

# Add noise to the data
X_test = add_noise(X_test, 0, 0.1)

# Define some constants
INPUT_SHAPE = (71,)  # Number of input features
NUM_CLASSES = 10  # Number of output classes (0-9)

# Load the ensemble of neural networks
ensemble_size = 10
ensemble = []

for i in range(ensemble_size):
    model = tf.keras.models.load_model(f'ensemble_model_{i}.h5')
    ensemble.append(model)

# Make predictions on the test set using the ensemble
y_test_pred_prob = np.zeros((len(X_test), NUM_CLASSES))
for model in ensemble:
    y_test_pred_prob += model.predict(X_test)

y_test_pred = np.argmax(y_test_pred_prob, axis=1)

# Write predicted labels to file
write_predictions_to_file(X_test, ensemble, 'testlabels.txt')
