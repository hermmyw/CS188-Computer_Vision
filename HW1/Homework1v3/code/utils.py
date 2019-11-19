import os
import cv2
import numpy as np
import time
import random
from sklearn import neighbors, svm, cluster, preprocessing


def load_data():
    test_path = '../data/test/'
    train_path = '../data/train/'

    train_classes = sorted([dirname for dirname in os.listdir(train_path)], key=lambda s: s.upper())
    test_classes = sorted([dirname for dirname in os.listdir(test_path)], key=lambda s: s.upper())
    train_labels = []
    test_labels = []
    train_images = []
    test_images = []
    for i, label in enumerate(train_classes):
        for filename in os.listdir(train_path + label + '/'):
            image = cv2.imread(train_path + label + '/' + filename, cv2.IMREAD_GRAYSCALE)
            train_images.append(image)
            train_labels.append(i)
    for i, label in enumerate(test_classes):
        for filename in os.listdir(test_path + label + '/'):
            image = cv2.imread(test_path + label + '/' + filename, cv2.IMREAD_GRAYSCALE)
            test_images.append(image)
            test_labels.append(i)

    return train_images, test_images, train_labels, test_labels


def KNN_classifier(train_features, train_labels, test_features, num_neighbors):
    # outputs labels for all testing images

    # train_features is an N x d matrix, where d is the dimensionality of the
    # feature representation and N is the number of training features.
    # train_labels is an N x 1 array, where each entry is an integer
    # indicating the ground truth category for each training image.
    # test_features is an M x d array, where d is the dimensionality of the
    # feature representation and M is the number of testing features.
    # num_neighbors is the number of neighbors for the KNN classifier

    # predicted_categories is an M x 1 array, where each entry is an integer
    # indicating the predicted category for each test image.

    neigh = neighbors.KNeighborsClassifier(n_neighbors=num_neighbors, algorithm='kd_tree', metric='euclidean')
    neigh.fit(train_features, train_labels)
    predicted_categories = neigh.predict(test_features)
    return predicted_categories


def SVM_classifier(train_features, train_labels, test_features, is_linear, svm_lambda):
    # this function will train a linear svm for every category (i.e. one vs all)
    # and then use the learned linear classifiers to predict the category of
    # every test image. every test feature will be evaluated with all 15 svms
    # and the most confident svm will "win". confidence, or distance from the
    # margin, is w*x + b where '*' is the inner product or dot product and w and
    # b are the learned hyperplane parameters.

    # train_features is an N x d matrix, where d is the dimensionality of
    # the feature representation and N the number of training features.
    # train_labels is an N x 1 array, where each entry is an integer 
    # indicating the ground truth category for each training image.
    # test_features is an M x d matrix, where d is the dimensionality of the
    # feature representation and M is the number of testing features.
    # is_linear is a boolean. If true, you will train linear SVMs. Otherwise, you 
    # will use SVMs with a Radial Basis Function (RBF) Kernel.
    # svm_lambda is a scalar, the value of the regularizer for the SVMs

    # predicted_categories is an M x 1 array, where each entry is an integer
    # indicating the predicted category for each test feature.

    test_labels = []

    total_bi_labels = []
    for i in range(15):
        temp_train_labels = []
        for label in train_labels:
            if label == i:
                temp_train_labels.append(0)
            else:
                temp_train_labels.append(1)
        total_bi_labels.append(temp_train_labels)
    total_bi_labels = np.array(total_bi_labels)

    if (is_linear == True):  # Linear
        clf = svm.LinearSVC(C=svm_lambda)
    else:  # Radial basis function kernel
        # 15 models
        for i in range(15):
            clf = svm.SVC(kernel='rbf', C=svm_lambda, gamma=10)

    clf.fit(train_features, train_labels)
    decision = clf.decision_function(test_features)  # n_samples * n_categories
    # decision = map(abs, decision)
    for i, conf in enumerate(decision):
        test_labels.append(np.argmax(conf))

    predicted_categories = test_labels


    return predicted_categories


def imresize(input_image, target_size):
    # resizes the input image, represented as a 2D array, to a new image of size [target_size, target_size]. 
    # Normalizes the output image to be zero-mean, and in the [-1, 1] range.

    # resizes the input image
    output_image = cv2.resize(input_image, (target_size, target_size),
                              interpolation=cv2.INTER_AREA)

    # normalizes the output image
    cv2.normalize(output_image, output_image, -1, 1, norm_type=cv2.NORM_MINMAX)
    n = cv2.mean(output_image)
    return output_image


def reportAccuracy(true_labels, predicted_labels):
    # generates and returns the accuracy of a model

    # true_labels is a N x 1 list, where each entry is an integer
    # and N is the size of the testing set.
    # predicted_labels is a N x 1 list, where each entry is an 
    # integer, and N is the size of the testing set. These labels 
    # were produced by your system.

    # accuracy is a scalar, defined in the spec (in %)

    correct = sum(1 for i, j in zip(true_labels, predicted_labels) if i == j)
    accuracy = correct / len(predicted_labels) * 100  # N != 0
    print("Accuracy: ", accuracy)
    return accuracy


def buildDict(train_images, dict_size, feature_type, clustering_type):
    # this function will sample descriptors from the training images,
    # cluster them, and then return the cluster centers.

    # train_images is a list of N images, represented as 2D arrays
    # dict_size is the size of the vocabulary,
    # feature_type is a string specifying the type of feature that we are interested in.
    # Valid values are "sift", "surf" and "orb"
    # clustering_type is one of "kmeans" or "hierarchical"

    # the output 'vocabulary' should be a list of length dict_size, with elements of size d, where d is the 
    # dimension of the feature. each row is a cluster centroid / visual word.

    # NOTE: Should you run out of memory or have performance issues, feel free to limit the 
    # number of descriptors you store per image.

    t0 = time.time()

    vocabulary = []
    descriptors = []
    n_features = 100

    print("BUILDING DICT:", feature_type, dict_size, clustering_type)
    if feature_type == 'sift':
        for i in train_images:
            sift = cv2.xfeatures2d.SIFT_create(n_features)
            kp, des = sift.detectAndCompute(i, None)
            if des is not None:
                for d in des:
                    descriptors.append(d)


    elif feature_type == 'surf':
        for i in train_images:
            surf = cv2.xfeatures2d.SURF_create(4000)
            kp, des = surf.detectAndCompute(i, None)
            # des = random.sample(list(des), n_features)
            if des is not None:
                for d in des:
                    descriptors.append(d)

    elif feature_type == 'orb':
        for i in train_images:
            orb = cv2.ORB_create(nfeatures=n_features)
            kp, des = orb.detectAndCompute(i, None)
            if des is not None:
                # des = random.sample(list(des), 5)
                for d in des:
                    descriptors.append(d)

    print("  DESCRIPTOR LENGTH:", len(descriptors))

    if clustering_type == 'kmeans':
        kmeans = cluster.KMeans(n_clusters=dict_size).fit(descriptors)
        vocabulary = kmeans.cluster_centers_

    if clustering_type == 'hierarchical':

        if descriptors is not None:
            descriptors = random.sample(list(descriptors), int(0.10*len(descriptors)))

        # clustering
        agg = cluster.AgglomerativeClustering(n_clusters=dict_size).fit(descriptors)
        labels = agg.labels_.astype(np.int32)

        # dict_size * feature_size 2D list
        vocabulary = [[0] * len(descriptors[0]) for i in range(dict_size)]
        len_vocab = [0] * dict_size

        # computing centroids (average) of the clusters
        for i in range(len(descriptors)):
            label = labels[i]
            vocabulary[label] = np.add(vocabulary[label], descriptors[i])
            len_vocab[label] = len_vocab[label] + 1
        for i, word in enumerate(vocabulary):
            vocabulary[i] = np.array(word / len_vocab[i])
        vocabulary = np.array(vocabulary)

    t1 = time.time()

    print("  BUILD DICT RUNTIME:", int(t1 - t0), "s")
    return vocabulary


def computeBow(image, vocabulary, feature_type):
    # extracts features from the image, and returns a BOW representation using a vocabulary

    # image is 2D array
    # vocabulary is an array of size dict_size x d
    # feature type is a string (from "sift", "surf", "orb") specifying the feature
    # used to create the vocabulary

    # BOW is the new image representation, a normalized histogram

    # Extract features from test image
    feature = 0

    # Build histogram dict_size*n_descriptors
    dict_size = len(vocabulary)
    histogram = [0] * len(vocabulary)
    MAX_FLOAT = np.finfo('float64').max
    label = 0

    if feature_type == 'sift':
        feature = cv2.xfeatures2d.SIFT_create()
    if feature_type == 'surf':
        feature = cv2.xfeatures2d.SURF_create()
    if feature_type == 'orb':
        feature = cv2.ORB_create()

    # Compute keypoints and descriptors for tested image
    kp, descriptor = feature.detectAndCompute(image, None)
    if descriptor is not None:
        if len(list(descriptor)) > 400:
            descriptor = random.sample(list(descriptor), 400)
        for d in descriptor:
            min_distance = MAX_FLOAT
            label = 0
            distance = np.array([])
            for i, word in enumerate(vocabulary):
                # compute distance between the feature found in the image to the feature in the vocab
                norm = np.linalg.norm(d - word)
                distance = np.append(distance, norm)
                # if norm < min_distance:
                #     min_distance = norm
                #     label = i
            label = np.argmin(distance)
            histogram[label] = histogram[label] + 1  # increment bucket

            # TODO: hamming distance for orb

        histogram = np.array(histogram) / len(descriptor)

    # hist, bin_edges = np.histogram(histogram, bins=range(0, dict_size+1), density=True)
    Bow = histogram

    return Bow  # type: nparray


def tinyImages(train_features, test_features, train_labels, test_labels):
    # Resizes training images and flattens them to train a KNN classifier using the training labels
    # Classifies the resized and flattened testing images using the trained classifier
    # Returns the accuracy of the system, and the overall runtime (including resizing and classification)
    # Does so for 8x8, 16x16, and 32x32 images, with 1, 3 and 6 neighbors

    # train_features is a list of N images, represented as 2D arrays
    # test_features is a list of M images, represented as 2D arrays
    # train_labels is a list of N integers, containing the label values for the train set
    # test_labels is a list of M integers, containing the label values for the test set

    # classResult is a 18x1 array, containing accuracies and runtimes, in the following order:
    # accuracies and runtimes for 8x8 scales, 16x16 scales, 32x32 scales
    # [8x8 scale 1 neighbor accuracy, 8x8 scale 1 neighbor runtime, 8x8 scale 3 neighbor accuracy, 
    # 8x8 scale 3 neighbor runtime, ...]
    # Accuracies are a percentage, runtimes are in seconds
    classResult = []
    for size in [8, 16, 32]:
        for number in [1, 3, 6]:
            # start timer
            t0 = time.time()

            # resize images
            train = [imresize(img, size) for img in train_features]
            test = [imresize(img, size) for img in test_features]
            # save in np arrays
            train = np.asarray(train).reshape(len(train), -1)
            test = np.asarray(test).reshape(len(test), -1)

            prediction = KNN_classifier(train, train_labels, test, number)
            accuracy = reportAccuracy(test_labels, prediction)

            classResult.append(accuracy)
            t1 = time.time()
            t = t1 - t0
            classResult.append(t)  # time in seconds

    return classResult
