#https://customers.pyimagesearch.com/lesson-sample-k-nearest-neighbor-classification/
#https://towardsdatascience.com/understanding-and-using-k-nearest-neighbours-aka-knn-for-classification-of-digits-a55e00cc746f
#https://www.tutorialspoint.com/scikit_learn/scikit_learn_kneighbors_classifier.htm
#https://www.kaggle.com/residentmario/kernels-and-support-vector-machine-regularization
#SVM_LINEAR
from tensorflow import keras
import numpy as np
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.svm import SVC
#Downloading mnist dataset

fashion_mnist = keras.datasets.fashion_mnist
(trImages, trLabels), (tImages, tLabels) = fashion_mnist.load_data()

#normalize
trImages = trImages.astype('float32')
tImages = tImages.astype('float32')
trImages=trImages/255
tImages=tImages/255

trImages = trImages.reshape(trImages.shape[0], trImages.shape[1] * trImages.shape[2]) #monodiastata
tImages = tImages.reshape(tImages.shape[0], tImages.shape[1] * tImages.shape[2])

accuracies = []

model = SVC(kernel='linear', decision_function_shape='ovr') # Linear Kernel
model.fit(trImages, trLabels) #fortwnoume to montelo

predictions = model.predict(tImages) #ksekinaei thb ekpaideush

# evaluate the model and update the accuracies list
score=model.score(tImages, tLabels)
print("accuracy=%.2f%%" % (metrics.accuracy_score(tLabels, predictions) * 100))
print("f1_score=%.2f%%" % (metrics.f1_score(tLabels, predictions, average= "weighted") * 100))
accuracies.append(score)

# show a final classification report demonstrating the accuracy of the classifier
# for each of the label
print("EVALUATION ON TESTING DATA")
print(classification_report(tLabels, predictions)) # matrix with our all-statistics