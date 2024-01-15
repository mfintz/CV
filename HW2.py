import scipy
import glob
import os
import numpy as np
import cv2
from scipy.cluster.vq import *
from sklearn.svm import LinearSVC
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


global true_test_labels


#
# This method returns a list of all filenames of images in the rootPath folder
# it assumes that DB images are stored in subfolders
#
def getImagesList(rootPath):
    image_list = []

    os.chdir(rootPath)
    for imageFileName in glob.glob("*.jpg"):
        image_list.append(rootPath + "\\" + imageFileName)
    os.chdir("..")
    return image_list


#
# This method returns an array of histograms (BOWs)
#
def getImagesBOWs(image_paths, extractor, quantization=[], k=100, iterations=3):
    des_list = []

    for image_path in image_paths:
        print("Reading image from " + image_path)
        img = cv2.imread(image_path)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = extractor.detectAndCompute(gray, None)
        des_list.append((image_path, des))

    #
    # The code below is taken from a snippet from the Internet - grouping
    # descriptiors in a stack
    #
    descriptors = des_list[0][1]
    for image_path, descriptor in des_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))

    #
    # Perform k-means clustering by building the vocabulary.
    # There is a way to pass a quantization as a parameter - this will be used
    # when processing the test images.
    # For the test images we want to use the same quantization which was
    # calculated on the DB images (aka training set)
    #
    if quantization == []:
        print("Start building the quantization")
        voc, variance = kmeans(descriptors, k, iterations)
        print("Quantization built")
    else:
        print("Using the preset quantization")
        voc = quantization

    #
    # Calculate the BOWs for each image
    # this small code is also take forn an Internet snippet
    #
    im_features = np.zeros((len(image_paths), k), "float32")
    for i in range(len(image_paths)):
        words, distance = vq(des_list[i][1], voc)
        for w in words:
            im_features[i][w] += 1

    #
    # Scaling was proposed in the Internet.  We may omit this, as it works
    # without it.
    #
    stdSlr = StandardScaler().fit(im_features)
    im_features = stdSlr.transform(im_features)

    return (im_features, voc)


#
# This method does the main work.
# 1.  It combines good and bad images into one long array
# 2.  It calculates BOWs for all DB images by using getImagesBOWs()
# 3.  It build LinearSVC based on the BOWs of the DB images
# 4.  It calculates BOWs of the test images BASED on the quantization of the DB
# images (this is very important)
# 5.  It runs the LinearSVC classifier adn returns its prediction
#
def RunDetector(difficultyLevel, bagSize=100, KMeansIiterations=10, testImagesDirectory=None):
    global prob
    sift = cv2.xfeatures2d.SIFT_create()

    #
    # first - read good images
    #
    image_paths = []
    if difficultyLevel == 1:
        image_paths += getImagesList("Easy")  # level 1

    if difficultyLevel == 2:
        image_paths += getImagesList("Normal")  # level 2

    if difficultyLevel == 3:
        image_paths += getImagesList("Hard")  # level 3

    if difficultyLevel == 4:
        image_paths += getImagesList("Easy")  # level 1
        image_paths += getImagesList("Normal")  # level 2
        image_paths += getImagesList("Hard")  # level 3

    # Mark good images as 1 since all the images till now were clocks

    image_classes = np.ones(len(image_paths), dtype=np.int)

    # Read bad images
    bad_images_paths = []
    if difficultyLevel == 1:
        bad_images_paths += getImagesList("Easy Not Clocks")

    if difficultyLevel == 2:
        bad_images_paths += getImagesList("Normal Not Clocks")

    if difficultyLevel == 3:
        bad_images_paths += getImagesList("Hard Not Clocks")

    if difficultyLevel == 4:
        bad_images_paths += getImagesList("Easy Not Clocks")
        bad_images_paths += getImagesList("Normal Not Clocks")
        bad_images_paths += getImagesList("Hard Not Clocks")

    # Mark bad (non-object) images as 0

    image_classes = np.append(image_classes, np.zeros(len(bad_images_paths), dtype=np.int))

    #
    # Process images - calculate BOWs
    #
    all_images_paths = np.append(image_paths, bad_images_paths)
    all_bows, all_quantization = getImagesBOWs(all_images_paths, sift, iterations=KMeansIiterations, k=bagSize)

    #
    # Train the Linear SVM (note, that image_classes contains 1s and 0s, e.g.
    # we have 2 classes only
    #
    clf = LinearSVC()
    clf.fit(all_bows, np.array(image_classes))

    #
    # Now we are ready to read test images.  Here we pass the same quantization
    # that we got from the calculation of DB images.
    # E.g.  we use the same visual words.
    #

    if testImagesDirectory == None:
        test_image_paths = getImagesList("Test positives")
        test_image_paths += getImagesList("Test negatives")  
    else:
        test_image_paths = getImagesList(testImagesDirectory)

    test_bows, unused_quantization = getImagesBOWs(test_image_paths, sift, quantization=all_quantization,
                                                   iterations=KMeansIiterations, k=bagSize)
    #
    # And the final stage - asking the classifier for a verdict
    #
    predictions = clf.predict(test_bows)
    prob = clf.decision_function(test_bows)

    return (test_image_paths, predictions)

#
# This routine may be used for checking the HW (as it was requested by Hagit
#
def ForChecker(testDirectory):
    os.chdir("database")

    #
    # RunDetector() method receives several parameters explained inn the comments in the function.
    # A short example: Running RunDetector(4, 100, 5, testDirectory);
    #       Will use all training folders (with clock, clock from strange point of view and some hard images. To use only straight clocks for training set the first parameter to 1
    #       Will use visual word dictionary of size 100
    #       Will run kmeans algorythm for 5 iterations (the more iteartions - the better is the quantization, but takes more time)
    #       Will use testDirectory subdirectory for test images
    #
    test_images, results = RunDetector(4, 100, 5, testDirectory);
    total = 0
    objects = 0

    for i in range(len(test_images)):
        imageName = test_images[i]
        total += 1

        if results[i] == 1:
            print(imageName + " - is an object")
            objects += 1
        else:
            print(imageName + " - is NOT an object")

    print("Percent of recognized objects is " + str(objects*100/total) + "%")
    os.chdir("..")
    return


#
# This is an example of how to run the routine for checking (uncomment the line)
# A directry named test should be created under the folder Database (this folder is a part of the submitted HW.
#
#ForChecker("test")

#
# Main
#
os.chdir("database")

#
# Parameters can be changed
# We leave it for the checker to play with the parameters (if there is a desire to it :)))
# We can't put here 10 iterations and 1000 bag size as it will take a lot of time for the checker to run the program
# Thus we put here small numbers for kmeans(), e.g. 100 and 3, although it the quality of object detection may decrease.
#
test_images, results = RunDetector(4, 100, 3)  # do it for each one (1=eas,2=normal,3=hard,4=all)


positive_test_image_paths = getImagesList("Test positives")
true_test_labels = np.ones(len(positive_test_image_paths), dtype=np.int)
negative_test_image_paths = getImagesList("Test negatives")
true_test_labels = np.append(true_test_labels,np.zeros(len(negative_test_image_paths), dtype=np.int))




decided_clock_true = 0
decided_clock_total = 0

for i in range(len(test_images)):
    imageName = test_images[i]
    if results[i] == 1:
        print(imageName + " - is a clock")
        decided_clock_total += 1
        if true_test_labels[i] == 1:
            decided_clock_true+=1
    else:
        print(imageName + " - is NOT a clock")

print("true:" + str(decided_clock_true))
print("clocks:"+ str(decided_clock_total))
print("precision is :" + str(decided_clock_true / decided_clock_total))
print("recall is :" + str(decided_clock_true / len(positive_test_image_paths)))

fpr, tpr, thresholds = roc_curve(true_test_labels, prob)
plt.figure()
plt.plot(fpr, tpr, label='ROC')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ALL TEST images')
plt.show()

