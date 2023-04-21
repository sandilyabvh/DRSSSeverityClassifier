# DRSS Severity Classifier

Diabetes is a metabolic disorder that affects blood sugar levels due to insufficient insulin production by the body. Diabetics have microvascular complications which are a subset of disorders that affect blood vessels in various organs. One such complication is Diabetic Retinopathy (DR), which occurs due to prolonged high blood sugar levels that cause blockage of the light-sensitive tissue at the back of the eye. In this paper, we discuss four unique machine learning-based classifier systems and various feature extraction techniques for Optical Coherence Tomography (OCT) images to classify DR using the Ocular Lesion Image Verification and Evaluation System (OLIVES) dataset.

The project aims to develop a Diabetic Retinopathy Severity Scale (DRSS) classifier using OCT images from the PRIME component of the OLIVES dataset. OCT images are high-resolution, non-invasive, cross-sectional images of biological tissues. Four distinct classifiers are implemented and evaluated based on balanced accuracy, F1 score, and other metrics. The dataset comprises 32,337 samples, divided into 658 patient volumes, with each volume containing 49 right (OD) and left (OS) eye images representing a single patient's clinical visit.



![UNetHeatMap_01_027_W52_OS_cn_final](https://user-images.githubusercontent.com/66162811/233683728-633b73d6-5edd-4b7b-89d7-7f2e0bba5ff0.png)

Running the Jupyter notebooks is self explanatory with relavant comments present.

The Case viewer GUI can be launched using the below steps:
1. conda activate ece8803
2. `python3 <repo path>/cases_viewer.py --data_root <parent directory containing prime test and train csvs> --annot_train_prime <path to df_prime_train.csv> --annot_test_prime <path to df_prime_test.csv>`

Directory Structure:
   |
   | -- Explainability: This directory contains relevant heat maps and Jupyter notebook to generate heat maps.
   | -- ProofRuns: This directory contains screencapture proofs at the time of running the classifier.
   | -- Uploading screen captures at the time of running classifier successfull.
   | -- 3DCNN.ipynb: Experimenting with a 3D-CNN implemented bycombining a ConvLSTM and a 2D-U-N…
   | -- Blending.ipynb: Contains the OCTDataset class that internally handles the blending logic. We run this OCTDataset class over the train and test data set.
        This is the method we follow when we have to generated a pre-processed image for each sample within a Volutme.
   | -- case_viewer.py: This is a GUI that helps in terms of visualization. One can scej
   | -- KNN.ipynb: Contains the code that initially reduces dimentions, flattens and later passes the inputs through KNN.
   | -- SVM.ipynb: Contains the code to the RBF Sampler and the SGDC based sVM. The RBF Sampler has contUNet-CNN.ipynb
   | -- NaiveBayes.ipynb: Contains the code to run Naive Bayes. We have to initialize the transforms first, run the OCTDataset and OCTCLassifier.
   | -- UNet-CNN.ipynb: Contains the U-Net classifier and dataset wrapper classes. To run the classifier, one must first train it on the train set after performing data augmentation like random rotations. Finally the model is run on the test set.
   | -- image_file_paths.txt: A text file listing the paths of unique images for the case_viewer.py to show when it is run. 
   
   
