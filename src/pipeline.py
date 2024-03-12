#!/usr/bin/python
# -*- coding: utf-8 -*-
 
import pandas as pd 
import numpy as np 
import matplotlib as plt 
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

def plot_categorical_relationships(column, df):
    # Plot the relationship between 'Transported' and column
    plt.figure(figsize=(8, 6))
    sns.countplot(x=column, hue='Transported', data=df)
    plt.title('Relatonship between Transported and ' + column)
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.show()

def manage_outilers(df, column, outlier_value=1000):
    # Get the mean for non-outlier values to replace 
    mean = df[df[column] < outlier_value][column].mean()
    # Get the number of outliers 
    df.loc[df[column] > outlier_value, column] = mean
    return df 

def preprocessing(df):
    df['HomePlanet'] = label_encoder.fit_transform(df['HomePlanet'])
    df['CryoSleep'] = label_encoder.fit_transform(df['CryoSleep'])
    df['VIP'] = label_encoder.fit_transform(df['VIP'])
    df['Destination'] = label_encoder.fit_transform(df['Destination'])

    df['HomePlanet'].replace(3, 1, inplace=True)
    df['CryoSleep'].replace(2, 0, inplace=True)
    df['VIP'].replace(2, 0, inplace=True)
    df['Destination'].replace(3, 2, inplace=True)
    encoded_planets_df = pd.get_dummies(df['HomePlanet'], prefix='Planet')
    encoded_destination_df = pd.get_dummies(df['Destination'], prefix='Destination')
    df = pd.concat([df, encoded_destination_df, encoded_planets_df], axis = 1)

    df = manage_outilers(df, 'RoomService', 1000)
    df = manage_outilers(df, 'FoodCourt', 2500)
    df = manage_outilers(df, 'ShoppingMall', 1000)
    df = manage_outilers(df, 'Spa', 1500)
    df = manage_outilers(df, 'VRDeck', 1500)

    columns_with_missing = df.columns[df.isnull().any()].tolist()
    columns_with_missing.remove('Name')
    columns_with_missing.remove('Cabin')

    for column in columns_with_missing:
        df[column].fillna(df[column].mean(), inplace=True)

    df['Cabin'].fillna('F/1/P', inplace=True)

    df['Side'] = df['Cabin'].str[-1]
    df['Deck'] = df['Cabin'].str[0]
    df['Num'] = df['Cabin'].str[2]

    df['Deck'] = label_encoder.fit_transform(df['Deck'])
    df['Groups'] = df['PassengerId'].str[:4].astype(int)
    df['Group_number'] = df['PassengerId'].str[6:7].astype(int)
    return df 
def feature_engineering(df):
    df['TotalCost'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']
    df['Is_spended'] = (df['TotalCost'] == 0).astype(int)
    df['Is_S_side'] = (df['Side'] == 'S').astype(int)
    df['From_to'] = df['HomePlanet'].astype(str) + '-' + df['Destination'].astype(str)
    df['From_to'] = label_encoder.fit_transform(df['From_to'])

    # Drop not needed columns 
    df = df.drop(columns=['PassengerId', 'Destination', 'HomePlanet', 'Side'])
    return df 