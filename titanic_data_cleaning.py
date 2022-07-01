# -*- coding: UTF-8 -*-
"""Cleaning data Titanic"""


import numpy as np
import pandas as pd


pd.options.display.max_columns = 12
# Import train dataset
train_df = pd.read_csv('train.csv')
print(train_df.head(10))
print(train_df.isnull().sum())
print(train_df.describe())
print(train_df.shape)
print(train_df.columns)

# Import test dataset
test_df = pd.read_csv('test.csv')
id_df = test_df['PassengerId']
print(test_df.head())
print(test_df.describe())
print(test_df.shape)
print(test_df.columns)

def prepare_data(df):
    """"Prepare train and test set for ML"""
    # Sex feature to numeric 0 or 1
    binary_gender = {'male': 0, 'female': 1}
    df['Sex'] = df['Sex'].map(binary_gender)
    # Make Family feature based on 'SibSp' and 'Parch'
    df['Family'] = df['SibSp'] + df['Parch']
    # Make numeric 'Cabin_indicator' feature based on 'Cabin' info
    df['Cabin_indicator'] = np.where(df['Cabin'].isnull(), 0, 1)
    # Drop insignificant features
    drop_feats = ['PassengerId', 'Name', 'Ticket', 'Embarked', 'SibSp', 'Parch', 'Cabin']
    df.drop(drop_feats, axis=1, inplace=True)
    # Fill NaN feature with its mean value
    for f in df.columns.values:
        df[f].fillna(df[f].mean(), inplace=True)
    return df

train_df_clean = prepare_data(train_df)

test_df_clean = prepare_data(test_df)

print("Train data_clean", train_df_clean.head(10), sep='\n')
print(train_df_clean.describe())
print(train_df_clean.info())
print(train_df_clean.shape)
print(train_df_clean.columns)
print(train_df_clean.isnull().sum())

print("\nTest data_clean", test_df_clean.head(10), sep='\n')
print(test_df_clean.describe())
print(test_df_clean.info())
print(test_df_clean.shape)
print(test_df_clean.columns)
print(test_df_clean.isnull().sum())
