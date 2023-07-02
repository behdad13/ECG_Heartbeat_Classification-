ECG classification with CNN (pipeline + model deployment with Flask)
==============================

Welcome to the comprehensive A-Z project that uses a Convolutional Neural Network (CNN) model for heartbeat classification.

In this project, I perform Exploratory Data Analysis (EDA), feature engineering, data modelling, model evaluation, and model deployment, each step described in detail with associated code in a Jupyter notebook. You can find answers to your questions in the **Notebooks** Directory. The project data can be found here: https://www.kaggle.com/datasets/shayanfazeli/heartbeat.


Below is an explanation of the Python scripts used in this project: Follow the steps to run the projects. 

1. First, install the required libraries by running the command ```pip install -r requirements.txt``` and create directories for **data** and its subfolders. Data will be downloaded automatically from the AWS S3 bucket. Because of the large data size, I can not upload **data** folder here. (sorry :(( )

2. ```make_dataset.py```: This script converts raw data into a more manageable format, which is then stored in the interim folder. Run the command in your IDE terminal ```python src/data/make_dataset.py data/raw data/interim```. 
* It is ```python src/data/make_dataset.py <path to your raw data> <path to your intermediate data>```.

3. ```build_features.py```: This script takes the data from the interim stage and processes it into a format ready for modelling. Run the command in your IDE terminal ```python src/features/build_features.py data/interim data/processed```.
* It is ```python src/features/build_features.py <path to your intermediate data> <path to your processed data>```.

4. ```train_model.py```: This script is responsible for training and validating the model. It includes hyperparameter tuning for optimal performance. Run the command in your IDE terminal ```python src/models/train_model.py data/processed/processed_train.csv models/```. 
* It is ```python src/models/train_model.py <path to your processed train data> <path to your model directory>```.

5. ```predict_model.py```: This script predicts outcomes and tests the model against the test and hold-out set.

* For testing testset, run the command in your IDE terminal: ```python -m src.models.predict_model data/processed/processed_test.csv models/best_model_filter_64_kernel_7_pool_2.h5 reports/figures/ test```.

* For testing holdout, run the command in your IDE terminal: ```python -m src.models.predict_model data/processed/processed_holdout.csv models/best_model_filter_64_kernel_7_pool_2.h5 reports/figures/ holdout```.
* it is ```python -m src.models.predict_model <path to your processed test/holdout data>  <path to your best model>  <path to your figure folder>  <name:test or holdout> ```.


6. ```visualize.py```: This script produces various visualizations such as graphs, charts, and confusion matrices to better understand the model's performance.
Run the command in your IDE terminal ```python src/visualization/visualize.py data/processed/processed_train.csv data/interim/inter_train.csv reports/figures/```.
* It is ```python src/visualization/visualize.py <path to your processed train data> <path to your intermediate train data> <path to your figure folder>```.




A dedicated folder named **deployment** is included in this project. It holds the Flask application associated with the project. The application accepts a .csv file containing ECG signal records and returns the corresponding ECG class type. Here is the Flask app for my project:

![image](https://github.com/behdad13/ECG_Heartbeat_Classification-/assets/58978680/2d30ddda-3e18-497d-be01-536be04d0728)

To deploy the model, run the command in your IDE terminal ```python deployment/app.py```. You can test the model with ```output.csv```, located in ```deployment->update``` directory.


To perform MLFlow analysis, you can run the following command in your IDE terminal: mlflow ui. This command will launch the MLFlow user interface, allowing you to access graphical analysis tools.

![image](https://github.com/behdad13/ECG_Heartbeat_Classification-/assets/58978680/d198d8f7-e72c-4a62-8e84-3c98460497c0)


Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── Deployement        <- Deployement of the model on the local host (Flask app)  
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
