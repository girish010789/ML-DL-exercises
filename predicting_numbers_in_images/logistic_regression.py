
##importing libraries
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

digits = load_digits()

##Print out total number of images and labels
print("Image data", digits.data.shape)
print("Label data", digits.target.shape)

##displaying some images using matplotlib
plt.figure(figsize=(20,4))
X = []
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
	plt.subplot(1,5,index+1)
	plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
	X.append(plt.title('Training data: %i\n' %label, fontsize = 20))

plt.show(X)


##split the data to train and test sets
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.23, random_state=2)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

##Train the model
LogisticRegr = LogisticRegression()
LogisticRegr.fit(x_train, y_train)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, max_iter=100, multi_class = 'ovr', n_jobs=1, penalty='12', random_state=None, solver='liblinear', tol=0.0001, verbose=0, warm_start=False)

##predict the model using test_set
print(LogisticRegr.predict(x_test[0].reshape(1,-1)))
LogisticRegr.predict(x_test[0:10])
predictions = LogisticRegr.predict(x_test)

##accuracy of the model
score = LogisticRegr.score(x_test, y_test)
print(score)

##print confusion matrix in a heat map
plt.figure(figsize=(9,9))
cm = metrics.confusion_matrix(y_test, predictions)
print(cm)
sns.heatmap(cm, annot=True, fmt=".3f", linewidth=.5, square=True, cmap="Blues_r");
plt.ylabel('Actual label')
plt.xlabel('predicted label')
all_sample_title = 'Accuracy score: {0}'.format(score)
plt.title(all_sample_title, size=15)
plt.show(all_sample_title)


##take some quick random samples to see how the predictions are done
index = 0
classifiedIndex = []
for predict, actual in zip(predictions, y_test):
	if predict==actual:
		classifiedIndex.append(index)
	index += 1
plt.figure(figsize=(20,3))
X_title = []
for plotIndex, wrong in enumerate(classifiedIndex[0:4]):
	plt.subplot(1,4,plotIndex +1)
	plt.imshow(np.reshape(x_test[wrong], (8,8)), cmap=plt.cm.gray)
	X_title.append(plt.title("predicted data: {}, Actual Data: {}".format(predictions[wrong], y_test[wrong]), fontsize=10))

plt.show(X_title)
	

