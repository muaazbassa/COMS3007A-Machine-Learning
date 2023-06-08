import numpy as np
import tensorflow as tf

# Load the ensemble model
ensemble = []
ensemble_size = 10

# Load each model in the ensemble
for i in range(ensemble_size):
    model_path = f'ensemble_model_{i}.h5'
    model = tf.keras.models.load_model(model_path)
    ensemble.append(model)

def getData():
    f = open("traindata.txt", "r")  # Update the file name to "testdata.txt"
    xstr = f.read()
    x = xstr.splitlines()
    x_vals_list = []
    for i in range(len(x)):
        xArr = x[i].split(',')
        tempArr = np.zeros(len(xArr))
        for j in range(len(xArr)):
            tempArr[j] = float(xArr[j])
        x_vals_list.append(np.array(tempArr))
    x_vals = np.array(x_vals_list)
    return x_vals

x_test = getData()

predictions_final = np.zeros(len(x_test), dtype=int)

# Make predictions on the test set using the ensemble
for model in ensemble:
    predictions = model.predict(x_test)
    predictions = tf.nn.softmax(predictions).numpy()
    predictions_argmax = np.argmax(predictions, axis=1)
    predictions_final += predictions_argmax

predictions_final = np.floor_divide(predictions_final, ensemble_size)
predictions_final = predictions_final.astype(int)

# Save the predictions to a file
with open("testlabels.txt", "w") as f:
    for prediction in predictions_final:
        f.write(str(prediction) + "\n")
