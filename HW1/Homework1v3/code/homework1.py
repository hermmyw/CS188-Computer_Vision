from utils import *
import argparse

from utils import tinyImages, load_data

parser = argparse.ArgumentParser(description='CS188.2 - Fall 19 - Homework 1')
parser.add_argument("--tiny", "-t", type=bool, default=False, help='run Tiny Images')
parser.add_argument("--create-path", "-cp", type=bool, default=True, help='create the Results directory')
args = parser.parse_args()

# The argument is included as an idea for debugging, with a few examples in the main. Feel free to modify it or add arguments.
# You are also welcome to disregard this entirely

#############################################################################################################################
# This file contains the main. All the functions you need to write are included in utils. You also need to edit the main.
# The main just gets you started with the data and highlights the high level structure of the project.
# You are free to modify it as you wish - the modifications you are required to make have been marked but you are free to make
# others.
# What you cannot modify is the number of files you have to save or their names. All the save calls are done for you, you
# just need to specify the right data.
#############################################################################################################################


if __name__ == "__main__":
    
    if args.create_path:
        # To save accuracies, runtimes, voabularies, ...
        if not os.path.exists('Results'):
            os.mkdir('Results') 
        SAVEPATH = 'Results/'
    
    # Load data, the function is written for you in utils
    train_images, test_images, train_labels, test_labels = load_data()
    
    if args.tiny:
        # You have to write the tinyImages function
        tinyRes = tinyImages(train_images, test_images, train_labels, test_labels)
    
        # Split accuracies and runtimes for saving  
        for element in tinyRes[::2]:
            # Check that every second element is an accuracy in reasonable bounds
            assert (7 < element and element < 20)
        acc = np.asarray(tinyRes[::2])
        runtime = np.asarray(tinyRes[1::2])
    
        # Save results as numpy arrays
        np.save(SAVEPATH + 'tiny_acc.npy', acc)
        np.save(SAVEPATH + 'tiny_time.npy', runtime)


    # Compute the Bow representation for the training and testing sets
    test_rep = [] # To store a set of BOW representations for the test images (given a vocabulary)
    train_rep = [] # To store a set of BOW representations for the train images (given a vocabulary)
    features = ['sift'] * 4 + ['surf'] * 4 + ['orb'] * 4 # Order in which features were used
    # for vocabulary generation

    # Create vocabularies, and save them in the result directory
    # You need to write the buildDict function
    vocabularies = []
    vocab_idx = [] # If you have doubts on which index is mapped to which vocabulary, this is referenced here
    # e.g vocab_idx[i] will tell you which algorithms/neighbors were used to compute vocabulary i
    # This isn't used in the rest of the code so you can feel free to ignore it

    t0_vocab = time.time()
    for feature in ['sift', 'surf', 'orb']:
        for algo in ['kmeans', 'hierarchical']:
            for dict_size in [20, 50]:
                vocabulary = buildDict(train_images, dict_size, feature, algo)
                filename = 'voc_' + feature + '_' + algo + '_' + str(dict_size) + '.npy'
                np.save(SAVEPATH + filename, np.asarray(vocabulary))
                vocabularies.append(vocabulary) # A list of vocabularies (which are 2D arrays)
                vocab_idx.append(filename.split('.')[0]) # Save the map from index to vocabulary

    t1_vocab = time.time()

    print("TIME TO BUILD", len(vocabularies), "DICT:", int((t1_vocab-t0_vocab)/60), "min")

    # A list to store the accuracies and one for runtimes
    # Use BOW features to classify the images with a KNN classifier
    knn_accuracies = []
    knn_runtimes = []
    # Use BOW features to classify the images with 15 Linear SVM classifiers
    lin_accuracies = []
    lin_runtimes = []
    # Use BOW features to classify the images with 15 Kernel SVM classifiers
    rbf_accuracies = []
    rbf_runtimes = []





    # You need to write ComputeBow()
    for i, vocab in enumerate(vocabularies):


        t0_bow = time.time()
        for image in train_images: # Compute the BOW representation of the training set
            rep = computeBow(image, vocab, features[i]) # Rep is a list of descriptors for a given image
            train_rep.append(rep)
        # np.save(SAVEPATH + 'bow_train_' + str(i) + '.npy', np.asarray(train_rep)) # Save the representations for vocabulary i

        print("   COMPLETE TRAIN BOW")

        for image in test_images: # Compute the BOW representation of the testing set
            rep = computeBow(image, vocab, features[i])
            test_rep.append(rep)
        # np.save(SAVEPATH + 'bow_test_' + str(i) + '.npy', np.asarray(test_rep)) # Save the representations for vocabulary i

        print("   COMPLETE TEST BOW")

        t1_bow = time.time()

        print("TIME TO BUILD BOW:", int(t1_bow-t0_bow), 's')


        train_rep_2D = np.array(train_rep).reshape(len(train_labels), -1)  # nparray N x D
        test_rep_2D = np.array(test_rep).reshape(len(test_labels), -1)  # nparray M x D

        # start timer


        # KNN_classifier
        t0 = time.time()
        knn_result = KNN_classifier(train_rep_2D, train_labels, test_rep_2D, 9)
        t1 = time.time()
        accuracy = reportAccuracy(test_labels, knn_result)
        runtime = t1 - t0
        knn_accuracies.append(accuracy)
        knn_runtimes.append(runtime)



        # SVM_classifier_linear
        t0 = time.time()
        lin_result = SVM_classifier(train_rep_2D, train_labels, test_rep_2D, True, 50.0)
        t1 = time.time()
        accuracy = reportAccuracy(test_labels, lin_result)
        runtime = t1 - t0
        lin_accuracies.append(accuracy)
        lin_runtimes.append(runtime)



        # SVM_classifier_rbf
        t0 = time.time()
        rbf_result = SVM_classifier(train_rep_2D, train_labels, test_rep_2D, False, 50.0)
        t1 = time.time()
        accuracy = reportAccuracy(test_labels, rbf_result)
        runtime = t1 - t0
        rbf_accuracies.append(accuracy)
        rbf_runtimes.append(runtime)


        train_rep = []  # reset the list to save the following vocabulary
        test_rep = [] # reset the list to save the following vocabulary
        
    


    np.save(SAVEPATH+'knn_accuracies.npy', np.asarray(knn_accuracies)) # Save the accuracies in the Results/ directory
    np.save(SAVEPATH+'knn_runtimes.npy', np.asarray(knn_runtimes)) # Save the runtimes in the Results/ directory

    np.save(SAVEPATH+'lin_accuracies.npy', np.asarray(lin_accuracies)) # Save the accuracies in the Results/ directory
    np.save(SAVEPATH+'lin_runtimes.npy', np.asarray(lin_runtimes)) # Save the runtimes in the Results/ directory

    np.save(SAVEPATH +'rbf_accuracies.npy', np.asarray(rbf_accuracies)) # Save the accuracies in the Results/ directory
    np.save(SAVEPATH +'rbf_runtimes.npy', np.asarray(rbf_runtimes)) # Save the runtimes in the Results/ directory
            
    