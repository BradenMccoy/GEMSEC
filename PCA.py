import csv
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plot

property_categorical = pd.read_csv('categorical_properties.csv', delimiter=',')
property_continuous = pd.red_csv('continuous_properties.csv', delimiter=',')

#read csvs into arrays

data_categorical = pd.DataFrame(columns=[property_categorical.], index=?) #which columns?

print(data_categorical.head())
print(data_categorical.shape())

scaled_data = preprocessing.minmax_scale(data_categorical) #transpose if the data is in columns

pca = PCA()
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)

# bar graphing
per_var = np.round(pca.explained_variance_ratio_*100, decimals = 1)
labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
plt.bar(x = range(1, len(per_var) + 1), height = per_var, tick_label = labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Categorical Scree plot')
plt.show()
#PCA graphing
#pca_df = pd.DataFrame(pca_data,index=[], columns = labels)
#plt.title('Categorical PCA')
#plt.xlabel('PC1 - {0}%'.format(per_var[0]))
#plt.ylabel('PC2 - {0}%'.format([per_var[1]]))
#for all pc's

#for sample in pca_df.index:
    #plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2)) #for all pc's