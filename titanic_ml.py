import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

print(df_train.head())

plt.scatter(df_train['Age'], df_train['Survived'])
