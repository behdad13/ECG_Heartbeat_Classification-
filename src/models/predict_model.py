from keras.models import load_model
from sklearn.metrics import classification_report
import os
import pandas as pd
from src.visualization.visualize import plot_confusion_matrix
import tensorflow.keras.utils as utils


# Prepare data for testing the model
def prepare_test_data(df_test):
    X_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1]

    # Reshape to be [samples][time steps][features]
    X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

    y_test_one = utils.to_categorical(y_test, num_classes=5)

    return X_test, y_test, y_test_one


# Evaluate the traine model with two data sets: 1.test set and hold-out
def evaluate_best_model(df_test, best_model_path, figure_name):
    X_test, y_test, y_test_one = prepare_test_data(df_test)

    # Load the saved model
    loaded_model = load_model(best_model_path)

    loss, accuracy = loaded_model.evaluate(X_test, y_test_one)

    y_pred = loaded_model.predict(X_test)
    y_pred_classes = y_pred.argmax(axis=-1)

    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)
    print(classification_report(y_test, y_pred_classes))
    plot_confusion_matrix(y_test, y_pred_classes, figure_name=figure_name)


os.environ['DIR_PATH'] = '/Users/behdad/sickkids_interview/ECG Heartbeat Categorization'
dir_path = os.getenv('DIR_PATH')

best_model_path = os.path.join(dir_path, 'models', 'best_model_filter_64_kernel_7_pool_2.h5')
df_test = pd.read_csv(os.path.join(dir_path, 'data/processed/processed_test.csv'))
df_holdout = pd.read_csv(os.path.join(dir_path, 'data/processed/processed_holdout.csv'))

#for running
evaluate_best_model(df_test, best_model_path, 'test')
evaluate_best_model(df_holdout, best_model_path, 'holdout')


# python src/models/predict_model.py