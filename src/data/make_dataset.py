# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split



@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    intermediate data ready to be processed (saved in ../interim).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Read the input CSV files
    df_train = pd.read_csv('https://ecg-heartbeat-dataset.s3.amazonaws.com/mitbih_train.csv', header=None)
    df_test = pd.read_csv('https://ecg-heartbeat-dataset.s3.amazonaws.com/mitbih_test.csv', header=None)


    df_train.to_csv(input_filepath + '/mitbih_train.csv')
    df_test.to_csv(input_filepath + '/mitbih_test.csv')

    # Perform your desired data processing steps here
    # Step 1: Create Hold-out dataset
    X_train = df_train.iloc[:, :-1]
    y_train = df_train[187]
    X_train, X_hold_out, y_train, y_hold_out = train_test_split(X_train,
                                                                y_train,
                                                                test_size=0.05,
                                                                random_state=42,
                                                                stratify=y_train)
    df_hold_out = pd.concat([X_hold_out, y_hold_out], axis=1)
    df_train = pd.concat([X_train, y_train], axis=1)

    # Save the processed data to the output filepath
    df_train.to_csv(output_filepath + '/inter_train.csv', index=False)
    df_hold_out.to_csv(output_filepath + '/inter_holdout.csv', index=False)
    df_test.to_csv(output_filepath + '/inter_test.csv', index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
