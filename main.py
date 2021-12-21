from numpy.core.numeric import cross
from utilsAA import *
from sklearn.utils import Bunch


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
table_x[:,8] = encode_feature(table_x[:,8]) #ExerciseAngina
table_x, attributes = one_hot_encode_feature(table_x, 2, attributes) #ChestPainType
table_x, attributes = one_hot_encode_feature(table_x, 5, attributes) #RestingECG
table_x, attributes = one_hot_encode_feature(table_x, 8, attributes) #ST_Slope

#create dictionary
heart_data = Bunch(data=table_x, target=table_y, data_names=attributes, target_names=classes)

#split data in train and test models
train_x, test_x, train_y, test_y = train_test_split(heart_data.data, heart_data.target, random_state=5)

#create dictionaries from train and test models
heart_train = Bunch(data=train_x, target=train_y, data_names=attributes, target_names=classes)
heart_test = Bunch(data=test_x, target=test_y, data_names=attributes, target_names=classes)

#get optimal min_samples_split
opt_split = 0
avg = 0
for i in range(2,100):
    spl = DecisionTreeClassifier(criterion='entropy', min_samples_split=i)
    scores = cross_val_score(spl, X=heart_data.data, y=heart_data.target,cv=10)
    if np.mean(scores) > avg:
        avg = np.mean(scores)
        opt_split = i

#get optimal min_samples_leaf
opt_leaf = 0
avg = 0
for i in range(1,21):
    spl = DecisionTreeClassifier(criterion='entropy', min_samples_split=opt_split, min_samples_leaf=i)
    scores = cross_val_score(spl, X=heart_data.data, y=heart_data.target,cv=10)
    if np.mean(scores) > avg:
        avg = np.mean(scores)
        opt_leaf = i

#get optimal max_depth
opt_depth = 0
avg = 0
for i in range(1,25):
    spl = DecisionTreeClassifier(criterion='entropy', min_samples_split=opt_split, min_samples_leaf=opt_leaf, max_depth=i)
    scores = cross_val_score(spl, X=heart_data.data, y=heart_data.target,cv=10)
    if np.mean(scores) > avg:
        avg = np.mean(scores)
        opt_depth = i

#create decision tree
dtc = DecisionTreeClassifier(criterion="entropy", min_samples_leaf= opt_leaf, min_samples_split= opt_split, max_depth= opt_depth)
dtc.fit(heart_train.data, heart_train.target)
print('Accuracy train:', dtc.score(heart_train.data, heart_train.target))
print('Accuracy test:', dtc.score(heart_test.data, heart_test.target))

#Normalize k-NN
scaler = StandardScaler()
scaler.fit(heart_data.data)
standard_heart_data = scaler.transform(heart_data.data)

scaler.fit(heart_train.data)
standard_heart_train = scaler.transform(heart_train.data)

scaler.fit(heart_test.data)
standard_heart_test = scaler.transform(heart_test.data)

#get optimal params
opt_n_neighbours = 0
opt_p = 0
avg = 0
for i in range(1,20):
    for j in range(1,20):
        clf = KNeighborsClassifier(n_neighbours=i, p=j)
        scores = cross_val_score(clf, X=standard_heart_data, y=heart_data.target,cv=10)
        if np.mean(scores) > avg:
            avg = np.mean(scores)
            opt_p = j
            opt_n_neighbours = i

#k-NN
clf = KNeighborsClassifier(n_neighbours=opt_n_neighbours, p=opt_p)
clf.fit(standard_heart_train,heart_train.target)
print('Accuracy train:', clf.score(standard_heart_train, heart_train.target))
print('Accuracy test:', clf.score(standard_heart_test, heart_test.target))
