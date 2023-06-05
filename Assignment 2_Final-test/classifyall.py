import numpy as np
import tensorflow as tf

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

# Load the saved model
model = tf.keras.models.load_model('model.h5')

# Make predictions on the test set
predictions = model.predict(x_test)
predictions = tf.nn.softmax(predictions).numpy()
predictions_final = np.argmax(predictions, axis=1)

# Save the predictions to a file
with open("testlabels.txt", "w") as f:
    for prediction in predictions_final:
        f.write(str(prediction) + "\n")
