import csv
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plot

property_matrix = pd.read_csv('properties.csv',delimiter=',')
property_matrix.set_index('Properties',inplace=True)
property_matrix = property_matrix.drop(columns=['id','dupe','S or P?', '6 Cluster', '8 Cluster', 'authors', 'Key', 'Prop'],axis=1).T.round(decimals=2)
property_categorical=pd.DataFrame(index=property_matrix.index,columns=property_matrix.columns)
for i in range(len(property_matrix.columns)):
    for j in range(len(property_matrix.index)):
        if (property_matrix.iloc[j,i]*10)%10==0 and (property_matrix.iloc[j,i]*100)%10==0:
            property_categorical.iloc[j,i]=property_matrix.iloc[j,i]
        else:
            property_categorical.iloc[j,i]=None

property_categorical=property_categorical.dropna(axis=1,how='any')

print(property_categorical.shape)
property_continuous=property_matrix[property_matrix.columns[~property_matrix.columns.isin(property_categorical.columns)]]
print(property_continuous.shape)
property_categorical.to_csv('categorical_properties.csv')
property_continuous.to_csv('continuous_properties.csv')