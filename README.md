Sickkids_Interview
==============================

Welcome to the comprehensive A-Z project that uses a Convolutional Neural Network (CNN) model for heartbeat classification.

In this project, I perform Exploratory Data Analysis (EDA), feature engineering, data modelling, model evaluation, and model deployment, each step described in detail with associated code in a Jupyter notebook. You can find these notebooks in the **Notebooks** Directory.

Below is an explanation of the Python scripts used in this project:

1. ```make_dataset.py```: This script converts raw data into a more manageable format, which is then stored in the interim folder. Run the command in your IDE terminal ```python src/data/make_dataset.py data/raw data/interim```.

2. ```build_features.py```: This script takes the data from the interim stage and processes it into a format ready for modelling. Run the command in your IDE terminal ```python src/features/build_features.py data/interim data/processed```.

3. ```train_model.py```: This script is responsible for training and validating the model. It includes hyperparameter tuning for optimal performance. Run the command in your IDE terminal ```python src/models/train_model.py data/processed/processed_train.csv models/```

4. ```predict_model.py```: This script predicts outcomes and tests the model against the test and hold-out set.

For testing testset, run the command in your IDE terminal: ```python -m src.models.predict_model data/processed/processed_test.csv models/best_model_filter_64_kernel_7_pool_2.h5 reports/figures/ test```.

For testing holdout, run the command in your IDE terminal: ```python -m src.models.predict_model data/processed/processed_holdout.csv models/best_model_filter_64_kernel_7_pool_2.h5 reports/figures/ holdout```.

5. ```visualize.py```: This script produces various visualizations such as graphs, charts, and confusion matrices to better understand the model's performance.
Run the command in your IDE terminal ```python src/visualization/visualize.py data/processed/processed_train.csv data/interim/inter_train.csv reports/figures/```.




A dedicated folder named **deployment** is included in this project. It holds the Flask application associated with the project. The application accepts a .csv file containing ECG signal records and returns the corresponding ECG class type. Here is the Flask app for my project:

![image](https://github.com/behdad13/Interview_Sick_kids/assets/58978680/8531b35a-8630-46f4-94f1-37903ce26595)

To deploy the model, run the command in your IDE terminal ```python deployment/app.py```





Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    │
    ├── Deployement        <- deployement of the model on the local host (Flask app)  
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
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
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
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
