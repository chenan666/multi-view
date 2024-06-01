import pandas as pd
import numpy as np
import random

def clean_and_save_data(inputfile: str, outputfile: str, encoding: str = 'utf-8'):
    # Read data
    df = pd.read_csv(inputfile, encoding=encoding)

    # Remove columns where all values are the same
    df_cleaned = df.loc[:, (df != df.iloc[0]).any()]

    # Drop specific columns if they exist
    if 'formula' in df_cleaned.columns:
        df_cleaned.drop(['formula'], axis=1, inplace=True)
    if 'target' in df_cleaned.columns:
        df_cleaned.drop(['target'], axis=1, inplace=True)

    # Calculate Pearson correlation matrix
    corr_matrix = df_cleaned.corr().abs()

    # Find columns with correlation > 0.7
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_candidates = [column for column in upper.columns if any(upper[column] > 0.7)]

    # Randomly select 20 features to drop
    to_drop = random.sample(to_drop_candidates, min(len(to_drop_candidates), 20))

    # Drop the selected columns
    df_cleaned = df_cleaned.drop(columns=to_drop)

    # Save the cleaned data
    df_cleaned.to_csv(outputfile, index=False)

    print("Data cleaning and saving done")

# Example usage:
inputfile = "data_path"
outputfile = "saved_data_path"
clean_and_save_data(inputfile, outputfile, encoding='gbk')
