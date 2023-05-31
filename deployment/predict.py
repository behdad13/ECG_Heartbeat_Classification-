import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import os



def get_prediction(file_name):

    X_test = pd.read_csv(file_name, index_col=False)
    #data preparation
    X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    os.environ['DIR_PATH'] = '/Users/behdad/sickkids_interview/ECG Heartbeat Categorization'
    dir_path = os.getenv('DIR_PATH')

    best_model_path = os.path.join(dir_path, 'models', 'best_model_filter_64_kernel_7_pool_2.h5')
    loaded_model = load_model(best_model_path)

    y_pred = loaded_model.predict(X_test)
    y_pred = y_pred.argmax(axis=-1)


    if int(y_pred) == 0:
        output = "Normal beat"
    elif int(y_pred) == 1:
        output = 'Supraventricular premature beat'
    elif int(y_pred) == 2:
        output = 'Premature ventricular contraction'
    elif int(y_pred) == 3:
        output = 'Fusion of ventricular and normal beat'
    else:
        output = 'Unclassifiable beat'

    return output

