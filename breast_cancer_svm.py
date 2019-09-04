### GET BREAST CANCER DATA FROM SKLEARN DATASET AND BUILD A MODEL

import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

##load breast cancer datasets
cancer = datasets.load_breast_cancer()

##print features and target names for cancer
#print(cancer.feature_names)
#print(cancer.target.names)
x = cancer.data
y = cancer.target

##split the dataset to train and test-sets
x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.2)
print(x_train,y_train)
classes = ['malignant','benign']
print(classes)

##running without any kernel function will give less prediction percentage.
##you can test with running poly function but it takes too much time.
##c = 2 will allow some soft errors in the hyperplane in svm
##degree is used when kernel is set to poly
#classifier = svm.SVC(kernel="linear",C=100)
classifier = svm.SVC(kernel="poly",degree=3)

##train the model
classifier.fit(x_train,y_train)

##predict the model with test data
y_predict = classifier.predict(x_test)

##accuracy of the model
accuracy = metrics.accuracy_score(y_test,y_predict)

##print accuracy of the model, if not very accurate, tune SVC parameters, Below is a good website to facilitate tuning of SVC parameters
##https://medium.com/all-things-ai/in-depth-parameter-tuning-for-svc-758215394769
print(accuracy)

##you can test with the k-nearest neighbors classification to compare accuracy with svm
##classifier = KNeighborsClassifier(n_neighbors=15)
