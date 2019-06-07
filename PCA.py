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
print(property_continuous.head())

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
pc = pca_data_continuous[:,0]
print(pc)
cont_dict = {}
i = 0
for j in amino_acids:
    cont_dict[pc[i]] = j
    i+= 1    
#for i in range(5):
    

print(cont_dict)

#print(pc1) #next step is to take each PC, and sort it ascending/descending order and convert that sorted list to its corresponding amino acid letter order.



# Categorical
per_var = np.round(pca_categorical.explained_variance_ratio_*100, decimals = 1)
labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
plt.bar(x = range(1, len(per_var) + 1), height = per_var, tick_label = labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Categorical Scree plot')
#plt.show()
#print(pca_data_categorical[:,0])

#PCA graphing
#pca_df = pd.DataFrame(pca_data_categorical,index=[], columns = labels)
#plt.title('Categorical PCA')
#plt.xlabel('PC1 - {0}%'.format(per_var[0]))
#plt.ylabel('PC2 - {0}%'.format([per_var[1]]))

#for sample in pca_df.index:
    #plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2)) #for all pc's

