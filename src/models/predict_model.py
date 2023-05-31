import click
from keras.models import load_model
from sklearn.metrics import classification_report
import pandas as pd
#from src.visualization.visualize import plot_confusion_matrix
import tensorflow.keras.utils as utils


def prepare_test_data(df_test):
    X_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1]
    X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
    y_test_one = utils.to_categorical(y_test, num_classes=5)

    return X_test, y_test, y_test_one


def evaluate_best_model(df_test, best_model_path, figure_name):
    X_test, y_test, y_test_one = prepare_test_data(df_test)
    loaded_model = load_model(best_model_path)
    loss, accuracy = loaded_model.evaluate(X_test, y_test_one)
    y_pred = loaded_model.predict(X_test)
    y_pred_classes = y_pred.argmax(axis=-1)

    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)
    print(classification_report(y_test, y_pred_classes))
    #plot_confusion_matrix(y_test, y_pred_classes, figure_name=figure_name)


@click.command()
@click.argument('df_test_path', type=click.Path(exists=True))
@click.argument('best_model_path', type=click.Path(exists=True))
@click.argument('figure_name', type=str)

def main(df_test_path, best_model_path, figure_name):
    df_test = pd.read_csv(df_test_path)
    evaluate_best_model(df_test, best_model_path, figure_name)


if __name__ == '__main__':
    main()