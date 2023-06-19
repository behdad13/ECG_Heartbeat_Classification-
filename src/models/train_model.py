import mlflow
import click
from keras.models import Sequential, save_model
from keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D
from keras.callbacks import EarlyStopping
import os
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow.keras.utils as utils


#prepare dataset for training deep learning model
def prepare_train_valid_data(df_train):
    df_train = shuffle(df_train)

    X_train = df_train.iloc[:, :-1]
    y_train = df_train.iloc[:, -1]

    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.20, random_state=42, stratify=y_train)

    X_train = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_validation = X_validation.values.reshape(X_validation.shape[0], X_validation.shape[1], 1)

    y_train = utils.to_categorical(y_train, num_classes=5)
    y_validation = utils.to_categorical(y_validation, num_classes=5)

    return X_train, X_validation, y_train, y_validation


#def shuffle_data(df):
#    return df.sample(frac=1, random_state=42)

def create_model(filter_size, kernel_size, pool_size):
    model = Sequential()

    # CNN model
    model.add(Conv1D(filters=filter_size, kernel_size=kernel_size, activation='relu', input_shape=(187,1)))
    model.add(Conv1D(filters=filter_size, kernel_size=kernel_size, activation='relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=pool_size))

    # Output layer
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(5, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def train_model(model, X_train, y_train, X_validation, y_validation):
    # Define early stopping callback
    early_stop = EarlyStopping(patience=3, monitor='val_loss', mode='min', verbose=1)

    # Train the model with validation data and early stopping
    model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1, validation_data=(X_validation, y_validation), callbacks=[early_stop])

    return model

def evaluate_model(model, X_validation, y_validation):
    # Evaluate the model on the validation set
    loss, accuracy = model.evaluate(X_validation, y_validation)
    return loss, accuracy


def save_best_model(model, model_path, filter_size, kernel_size, pool_size):
    model_path = os.path.join(model_path, f"best_model_filter_{filter_size}_kernel_{kernel_size}_pool_{pool_size}.h5")
    save_model(model, model_path)
    return model_path


def log_metrics(filter_size, kernel_size, pool_size, accuracy, loss):
    mlflow.log_params({"filter_size": filter_size, "kernel_size": kernel_size, "pool_size": pool_size})
    mlflow.log_metrics({"accuracy": accuracy, "loss": loss})


@click.command()
@click.argument('df_train_path', type=click.Path(exists=True))
@click.argument('model_path', type=click.Path())
def main(df_train_path, model_path):
    # Define the hyperparameter values
    filter_sizes = [64, 128]
    kernel_sizes = [3, 5, 7]
    pool_size = 2

    # Create an experiment in MLflow
    mlflow.set_experiment("Hyperparameter_Tuning")

    # Initialize variables to track the best model
    best_accuracy = 0.0
    best_model_path = ""

    # Load the df_train from the provided path
    df_train = pd.read_csv(df_train_path)

    for filter_size in filter_sizes:
        for kernel_size in kernel_sizes:
            with mlflow.start_run():
                X_train, X_validation, y_train, y_validation = prepare_train_valid_data(df_train)

                model = create_model(filter_size, kernel_size, pool_size)

                model = train_model(model, X_train, y_train, X_validation, y_validation)

                loss, accuracy = evaluate_model(model, X_validation, y_validation)

                # Save the model if it has the highest accuracy so far
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_path = save_best_model(model, model_path, filter_size, kernel_size, pool_size)

                log_metrics(filter_size, kernel_size, pool_size, accuracy, loss)

    # Print the path to the best model
    print("Best model saved at:", best_model_path)

    # Fetch the best run based on the highest accuracy
    best_run = mlflow.search_runs(order_by=['metrics.accuracy DESC']).iloc[0]

    # Access the best hyperparameters and metrics
    best_kernel_size = best_run['params.kernel_size']
    best_pool_size = best_run['params.pool_size']
    best_filter_size = best_run['params.filter_size']
    best_accuracy = best_run['metrics.accuracy']

    print("Best Hyperparameters:")
    print("kernel_size:", best_kernel_size)
    print("filter_size:", best_filter_size)
    print("pool_size:", best_pool_size)
    print("Best Accuracy:", round(best_accuracy, 4))

if __name__ == '__main__':
    main()
