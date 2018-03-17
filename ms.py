import numpy as np
import sklearn
from sklearn import svm
from sklearn.datasets import load_files
from sklearn.externals import joblib
from sklearn.svm import SVC
import csv


clf = svm.SVC(decision_function_shape='ovo')

musicdir = '/home/shijo/Desktop/genreclass/genres'

clf = load_files(musicdir, shuffle=True)

#print(clf.data[0][0:50])


str = clf.data

#print(str[0])
X =[X.split(',') for X in str[0]]
#print(X[0])
#print(type(X))
k = [repr(l) for z in X for l in z]
print(type(k))


#y = clf.target

#clf.fit(X,y)

'''import numpy as np
from os import listdir

directory_path = '/home/shijo/Desktop/genreclass/genres/blues'
file_types = ['npy', 'npz']

np_vars = {dir_content: np.load(dir_content)
           for dir_content in listdir(directory_path)
           if dir_content.split('.')[-1] in file_types}





#


#print(clf.filenames[0])
#print (clf.data[0])

#from sklearn.feature_extraction import DictVectorizer
#vec = DictVectorizer(dtype=float,sparse=True)
#X = vec.fit_transform(clf.data).toarray()
#print (clf.data[1])
#y = clf.target
#clf.fit(X, y)
#X = [float(i) for i in clf.data.split(',')]
#data = np.genfromtxt(clf.data, dtype=float, delimiter=',', names=True)
#print(type(data))

dt = np.load("/home/shijo/Desktop/genreclass/genres/blues")
print(type(dt))


SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

joblib.dump(clf,'/home/shijo/Desktop/class/data.pkl')
joblib.dump(clf, '/home/shijo/Desktop/class/model.pkl')'''
