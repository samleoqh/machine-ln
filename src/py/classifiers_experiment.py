# Compare 9 classifiers performance by accuracy on cross-validation and
# by precision on fixed additional validation data
"""
directory structure:
```
dataset/
    train/
        Type_1/
            001.jpg
            002.jpg
            ...
        Type_2/
            001.jpg
            002.jpg
            ...
    validation/
        Type_1/
            001.jpg
            002.jpg
            ...
        Type_2/
            001.jpg
            002.jpg
            ...
```
"""
# import the necessary packages
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os


def extract_color_histogram(image, bins=(8, 8, 8)):
    # extract a 3D color histogram from the HSV color space using
    # the supplied number of `bins` per channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])

    # handle normalizing the histogram if we are using OpenCV 2.4.X
    if imutils.is_cv2():
        hist = cv2.normalize(hist)

    # otherwise, perform "in place" normalization in OpenCV 3
    else:
        cv2.normalize(hist, hist)

    # return the flattened histogram as the feature vector
    return hist.flatten()


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--train", type=str, default='cervix/train',
                help="path to input dataset")
ap.add_argument("-v", "--test", type=str, default='cervix/validation',
                help="path to input dataset")
args = vars(ap.parse_args())

# grab the list of images that we'll be describing
print("Describing images...")
imageTrainPaths = list(paths.list_images(args["train"]))
imageTestPaths = list(paths.list_images(args["test"]))

# initialize the data matrix and labels list
data = []
labels = []

data_test = []
labels_test = []

# loop over the input train images
for (i, imagePath) in enumerate(imageTrainPaths):
    # load the image and extract the class label
    image = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-2]

    # extract a color histogram from the image, then update the
    # data matrix and labels list
    hist = extract_color_histogram(image)
    data.append(hist)
    labels.append(label)

    # show an update every 1,000 images
    if i > 0 and i % 100 == 0:
        print("Processed {}/{}".format(i, len(imageTrainPaths)))

# loop over the input train images
for (i, imagePath) in enumerate(imageTestPaths):
    # load the image and extract the class label
    image = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-2]

    # extract a color histogram from the image, then update the
    # data matrix and labels list
    hist = extract_color_histogram(image)
    data_test.append(hist)
    labels_test.append(label)

    # show an update every 1,000 images
    if i > 0 and i % 100 == 0:
        print("Processed {}/{}".format(i, len(imageTestPaths)))

# encode the labels, converting them from strings to integers
le = LabelEncoder()
labels = le.fit_transform(labels)
labels_test = le.fit_transform(labels_test)

# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
#print("[INFO] constructing training/testing split...")
#(trainData, testData, trainLabels, testLabels) = train_test_split(
#    np.array(data), labels, test_size=0.25, random_state=42)

X_train = np.array(data)
y_train = labels
X_test = np.array(data_test)
y_test = labels_test


names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "SGDClassifier",
         #"Gaussian Process",
         "Decision Tree", "Random Forest", "MLPClassifier", "AdaBoost",
         "Naive Bayes"]

classifiers = [
    KNeighborsClassifier(59),
    LinearSVC(),
    SVC(gamma=2, C=1),
    SGDClassifier(loss="log", n_iter=10),
    #GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=15),
    RandomForestClassifier(n_estimators=100, max_features='sqrt'),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(learning_rate=0.1),
    GaussianNB()]

# closs_validation accuracy experiments
from sklearn.model_selection import cross_val_score
results = {}
for name, clf in zip(names, classifiers):
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    results[name] = scores

for name, scores in results.items():
    print("%20s | Accuracy: %0.2f%% (+/- %0.2f%%)" % (name, 100 * scores.mean(), 100 * scores.std() * 2))


# iterate over classifiers by using fixed additional validation data
for name, model in zip(names, classifiers):
    print("Training and evaluating classifier {}".format(name))
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions, target_names=le.classes_))
