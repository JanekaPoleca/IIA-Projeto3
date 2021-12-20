from utilsAA import *
from sklearn.utils import Bunch


from sklearn.datasets import load_iris, load_breast_cancer # conjuntos de dados
from sklearn.tree import DecisionTreeClassifier, plot_tree # árvore de decisão
from sklearn.neighbors import KNeighborsClassifier # k-NN
from sklearn.model_selection import train_test_split, cross_val_score # cross-validation
from sklearn.preprocessing import StandardScaler # normalização dos atributos
import numpy as np 
import matplotlib.pyplot as plt # gráficos
from utilsAA import * # módulo distribuido com o guião com funções auxiliares




#load csv
table_x, table_y, attributes, classes = load_data("heart.csv")

#remove id
table_x = table_x[:,1:]
attributes = attributes[1:]

#encode string attributes
table_x[:,1] = encode_feature(table_x[:,1]) #Sex
table_x[:,2] = encode_feature(table_x[:,2]) #ChestPainType
table_x[:,6] = encode_feature(table_x[:,6]) #RestingECG
table_x[:,8] = encode_feature(table_x[:,8]) #ExerciseAngina
table_x[:,10] = encode_feature(table_x[:,10]) #ST_Slope

#create dictionary
heart_data = Bunch(data=table_x, target=table_y, data_names=attributes, target_names=classes)

#split data in train and test models
train_x, test_x, train_y, test_y = train_test_split(heart_data.data, heart_data.target, random_state=5)

#create dictionaries from train and test models
heart_train = Bunch(data=train_x, target=train_y, data_names=attributes, target_names=classes)
heart_test = Bunch(data=test_x, target=test_y, data_names=attributes, target_names=classes)

#create decision tree
dtc = DecisionTreeClassifier(criterion="entropy", max_depth=None, min_samples_split=2, min_samples_leaf=1)

dtc.fit(train_x, train_y)

dtc.score(test_x, test_y)



#Draw decision tree (not working??)
plt.figure(figsize=[20,15])
plot_tree(dtc, feature_names=heart_train.data_names, class_names=heart_train.target_names, filled=True, rounded=True)
plt.show()


scores = cross_val_score(dtc,
                         X=heart_train.data,
                         y=heart_train.target,
                         cv=10
                        )