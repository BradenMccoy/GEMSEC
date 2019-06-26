import csv
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt

amino_acids = ("a", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "p", "q", "r", "s", "t", "v", "w", "y")

property_categorical = pd.read_csv('categorical_properties.csv', delimiter=',')
property_continuous = pd.read_csv('continuous_properties.csv', delimiter=',')

property_categorical.set_index('Unnamed: 0',inplace=True)
property_continuous.set_index('Unnamed: 0',inplace=True)

#print(property_categorical.head())
#print(property_continuous.head())

scaled_data_categorical = preprocessing.minmax_scale(property_categorical)
scaled_data_continuous = preprocessing.scale(property_continuous)
pca_categorical = PCA()
pca_continuous = PCA()
pca_continuous.fit(scaled_data_continuous)
pca_categorical.fit(scaled_data_categorical)
pca_data_continuous = pca_continuous.transform(scaled_data_continuous)
pca_data_categorical = pca_categorical.transform(scaled_data_categorical)

# Continuous
per_var = np.round(pca_continuous.explained_variance_ratio_*100, decimals = 1)
labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
plt.bar(x = range(1, len(per_var) + 1), height = per_var, tick_label = labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Continuous Scree plot')
#plt.show()
print("\n CONTINUOUS")
for i in range(5):
    pc = pca_data_continuous[:,i]
    print("\n pc" + str(i + 1))
    #print(pc)
    cont_dict = {}
    j = 0
    for k in amino_acids:
        cont_dict[pc[j]] = k
        j+= 1    
    ordered_pc = ""
    print("\nNormal:")
    for key, value in sorted(cont_dict.items(), reverse = True):
        #print(key, value)
        #print("\n")
        ordered_pc += value
    print(ordered_pc)
    print("\n")
    print("\nReversed:")
    ordered_pc = ""
    for key, value in sorted(cont_dict.items()):
        #print(key, value)
        #print("\n")
        ordered_pc += value
    print(ordered_pc)


print("\n")

# Categorical
per_var = np.round(pca_categorical.explained_variance_ratio_*100, decimals = 1)
labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
plt.bar(x = range(1, len(per_var) + 1), height = per_var, tick_label = labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Categorical Scree plot')
print("\n CATEGORICAL")
for i in range(5):
    pc = pca_data_categorical[:,i]
    print("\n pc" + str(i + 1))
    
    cat_dict = {}
    j = 0
    for k in amino_acids:
        cat_dict[pc[j]] = k
        j+= 1    
    print("\nNormal:")
    ordered_pc = ""
    for key, value in sorted(cat_dict.items(), reverse = True):
        #print(key, value)
        #print("\n")
        ordered_pc += value
    
    print(ordered_pc)
    print("\nReversed:")
    ordered_pc = ""
    for key, value in sorted(cat_dict.items()):
        #print(key, value)
        #print("\n")
        ordered_pc += value
    print(ordered_pc)
#plt.show()