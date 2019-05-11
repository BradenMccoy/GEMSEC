import csv
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plot

property_categorical = pd.read_csv('categorical_properties.csv', delimiter=',')
property_continuous = pd.red_csv('continuous_properties.csv', delimiter=',')

