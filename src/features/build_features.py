import pandas as pd
from sklearn.utils import resample
import click
import logging

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Read the input CSV files
    df_train = pd.read_csv(input_filepath + '/inter_train.csv', header=None)
    df_hold_out = pd.read_csv(input_filepath + '/inter_train.csv')
    df_test = pd.read_csv(input_filepath + '/inter_test.csv')

    # Perform your desired data processing steps here
    # Balance the Classes in train set
    df_train_0 = df_train[df_train[187] == 0]
    df_train_1 = df_train[df_train[187] == 1]
    df_train_2 = df_train[df_train[187] == 2]
    df_train_3 = df_train[df_train[187] == 3]
    df_train_4 = df_train[df_train[187] == 4]

    df_1_upsample = resample(df_train_1, n_samples=10000, replace=True, random_state=123)
    df_2_upsample = resample(df_train_2, n_samples=20000, replace=True, random_state=123)
    df_3_upsample = resample(df_train_3, n_samples=6000, replace=True, random_state=123)
    df_4_upsample = resample(df_train_4, n_samples=15000, replace=True, random_state=123)
    df_0_downsample = df_train_0.sample(n=30000, random_state=123)

    df_train_balanced = pd.concat([df_0_downsample, df_1_upsample, df_2_upsample, df_3_upsample, df_4_upsample])

    # Save the processed data to the output filepath
    df_train_balanced.to_csv(output_filepath + '/processed_train.csv', index=False)
    df_hold_out.to_csv(output_filepath + '/processed_holdout.csv', index=False)
    df_test.to_csv(output_filepath + '/processed_test.csv', index=False)

    logger.info('Data processing complete.')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()


