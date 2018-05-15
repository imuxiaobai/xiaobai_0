＃xiaobai_0
import numpy as py
import pandas as pd
from pandas import Series, DataFrame

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Imputer

train_data = pd.read_csv('./data/train.csv')
train_data.head()

train_data.describe()

sns.barplot(x='Sex',y='Survived',data=train_data)
#<matplotlib.axes._subplots.AxesSubplot at 0x16501a1ce10>
plt.show()

sns.barplot(x='Embarked',y='Survived',hue='Sex',data=train_data)
plt.show()

sns.pointplot(x='Pclass',y='Survived',hue='Sex',data=train_data,
             palette={'male':'blue','female':'pink'},
             markers=['*','o'],linestyles=['-','--'])
plt.show()

grid = sns.FacetGrid(train_data,col='Survived',row='Sex',
                    size=2.2,aspect=1.6)
grid.map(plt.hist,'Age',alpha=.5,bins=20)
grid.add_legend()
plt.show()

sns.barplot(x='SibSp',y='Survived',data=train_data)
plt.show()

sns.barplot(x='Parch',y='Survived',data=train_data)
plt.show()

train_data.Sex.unique()
array(['male', 'female'], dtype=object)
train_data.loc[train_data.Sex == 'male','Sex'] = 1
train_data.loc[train_data.Sex == 'female','Sex'] = 0
train_data.head()
