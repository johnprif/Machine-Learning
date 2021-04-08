import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import math
from sklearn.neighbors import KNeighborsClassifier

fashion_mnist = keras.datasets.fashion_mnist
(trImages, trLabels), (tImages, tLabels) = fashion_mnist.load_data()

print("--------------------------")
print("Dimensions of Train Set")
print("Dimension(trImages)=",np.shape(trImages))
print("There are", np.shape(trImages)[0], "images where each image is", np.shape(trImages)[1:], "in size")
print("There are", np.shape(np.unique(tLabels))[0], "unique image labels")
print("--------------------------")
print("Dimensions of Test Set")
print("Dimension(tImages)=",np.shape(tImages), "Dimension(tLabels)=", np.shape(tLabels)[0])
print("--------------------------")

paramk = 11  # parameter k of k-nearest neighbors
trImages = trImages.astype('float32')
tImages = tImages.astype('float32')
trImages=trImages/255
tImages=tImages/255

trImages = trImages.reshape(trImages.shape[0], trImages.shape[1] * trImages.shape[2])
tImages = tImages.reshape(tImages.shape[0], tImages.shape[1] * tImages.shape[2])

knn = KNeighborsClassifier(paramk)
knn.fit(trImages, trLabels)
y_pred_knn = knn.predict(tImages)

numTrainImages =600# np.shape(trLabels)[0]  # so many train images
numTestImages =600# np.shape(tLabels)[0] # so many test images
counter = 0
arrayKNNLabels = np.array([])
for iTeI in range(1, numTestImages):
    arrayL2Norm = np.array([])  # store distance of a test image from all train images
    for jTrI in range(numTrainImages):
        #Euclidean distance type function
        l2norm = np.sum(((trImages[jTrI] - tImages[iTeI])) ** 2) ** (
            0.5)  # distance between two images; 255 is max. pixel value ==> normalization
        arrayL2Norm = np.append(arrayL2Norm, l2norm)

    sIndex = np.argsort(arrayL2Norm)  # sorting distance and returning indices that achieves sort, saves the smallest distance

    kLabels = trLabels[sIndex[0:paramk]]  # choose first k labels
    (values, counts) = np.unique(kLabels, return_counts=True)  # find unique labels and their counts
    arrayKNNLabels = np.append(arrayKNNLabels, values[np.argmax(counts)])
    print(arrayL2Norm[sIndex[0]], kLabels, arrayKNNLabels[-1], tLabels[iTeI])
    #print(arrayL2Norm.astype(int))
    #accuracy_score(arrayKNNLabels.astype(int), arrayL2Norm.astype(int))
    #if not me then show my neightbours(geitones)
    if arrayKNNLabels[-1] != tLabels[iTeI]:

        plt.figure(1)
        plt.imshow(tImages[iTeI])
        plt.draw()

        for i in range(numTrainImages):
            if trLabels[i] == arrayKNNLabels[-1]:
                counter += 1
                plt.figure(2)
                plt.imshow(trImages[i])
                plt.draw()
                break

        plt.show()
    # elif arrayKNNLabels[-1] == tLabels[iTeI]:
    #     #counter +=1
    #     continue
    #x = *(numTestImages-counter)/numTestImages
accuracy = tf.metrics.accuracy_score(tImages , y_pred_knn)
print(counter)
print("The accurancy is: ",accuracy )
#print(arrayKNNLabels.astype(int))


