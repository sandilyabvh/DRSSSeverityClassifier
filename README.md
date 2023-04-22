# DRSS Severity Classifier

Diabetes is a metabolic disorder that affects blood sugar levels due to insufficient insulin production by the body. Diabetics have microvascular complications which are a subset of disorders that affect blood vessels in various organs. One such complication is Diabetic Retinopathy (DR), which occurs due to prolonged high blood sugar levels that cause blockage of the light-sensitive tissue at the back of the eye. In this paper, we discuss four unique machine learning-based classifier systems and various feature extraction techniques for Optical Coherence Tomography (OCT) images to classify DR using the Ocular Lesion Image Verification and Evaluation System (OLIVES) dataset.

The project aims to develop a Diabetic Retinopathy Severity Scale (DRSS) classifier using OCT images from the PRIME component of the OLIVES dataset. OCT images are high-resolution, non-invasive, cross-sectional images of biological tissues. Four distinct classifiers are implemented and evaluated based on balanced accuracy, F1 score, and other metrics. The dataset comprises 32,337 samples, divided into 658 patient volumes, with each volume containing 49 right (OD) and left (OS) eye images representing a single patient's clinical visit.



![UNetHeatMap_01_027_W52_OS_cn_final](https://user-images.githubusercontent.com/66162811/233683728-633b73d6-5edd-4b7b-89d7-7f2e0bba5ff0.png)


## Case View GUI

The Case viewer GUI can be launched using the below steps:
1. conda activate ece8803
2. `python3 <repo path>/cases_viewer.py --data_root <parent directory containing prime test and train csvs> --annot_train_prime <path to df_prime_train.csv> --annot_test_prime <path to df_prime_test.csv>`

## Directory Structure
   ```
├── cases_viewer.py: This is a GUI that helps in terms of visualization.
├── Explainability: This directory contains relevant heat maps and Jupyter notebook to generate heat maps.
│   ├── 0_grad_cam_heatmap.jpg
│   ├── 0_original_image.jpg
│   ├── 1_grad_cam_heatmap.jpg
│   ├── 1_original_image.jpg
│   ├── 2_grad_cam_heatmap.jpg
│   ├── 2_original_image.jpg
│   ├── 3_grad_cam_heatmap.jpg
│   ├── 3_original_image.jpg
│   ├── 4_grad_cam_heatmap.jpg
│   ├── 4_original_image.jpg
│   ├── 5_grad_cam_heatmap.jpg
│   ├── 5_original_image.jpg
│   ├── 6_grad_cam_heatmap.jpg
│   ├── 6_original_image.jpg
│   ├── 7_grad_cam_heatmap.jpg
│   ├── 7_original_image.jpg
│   ├── 8_grad_cam_heatmap.jpg
│   ├── 8_original_image.jpg
│   ├── CNN_Test_Latest.ipynb
│   ├── ImageOrder.txt
│   └── U-Net_CNN_GradCAMRunScreenCapture.pdf
├── image_file_paths.txt: A text file listing the paths of unique images for the case_viewer.py to show when it is run. 
├── ProofRuns: This directory contains screencapture proofs at the time of running the classifier.
│   ├── KNN_Accuracy_45_73.pdf
│   ├── Naive_Bayes_Accuracy_0_4955.pdf
│   ├── SVM_Screenshot.pdf
│   └── U-Net_CNN_RunScreenCapture.pdf
├── Python_Notebooks
│   ├── 3DCNN.ipynb: Experimental code with a 3D-CNN implemented by combining a ConvLSTM and a 2D-U-Net.
│   ├── Blending.ipynb: Helper script containing the OCTDataset class that internally handles different types of OCT image blending. We run this OCTDataset class over the train and test data set. This is the method we follow when we have to generated a pre-processed image for each sample within a volume.
│   ├── KNN.ipynb: Contains the code that initially reduces dimensions, flattens and later passes the inputs through KNN.
│   ├── NaiveBayes.ipynb: Contains the code to run Naive Bayes. We have to initialize the transforms first, run the OCTDataset and OCTCLassifier.
│   ├── SVM.ipynb: Contains the code to the RBF Sampler and the SGDC based sVM. The RBF Sampler has contUNet-CNN.ipynb
│   └── UNet-CNN.ipynb: Contains the U-Net classifier and dataset wrapper classes. To run the classifier, one must first train it on the train set after performing data augmentation like random rotations. Finally the model is run on the test set.
├── FML_Term_Paper.pdf: Project report in IEEE two column format.
├── README.md

   ```
## Requirements
The entire project was developed on Google Colab with PRIME dataset stored in Google drive and mounted before each run into the Google Colab session.
1. PRIME dataset and CSV files listing train and test dataset in Google drive.
2. Provide Python notebook uploaded to Google Colab and Runtime session must be set to GPU. 
3. Highly recommend purchasing premium compute units for better reliability.

## Pre-processing images
1. Open `Blending.ipynb` notebook in Google Colab and run the first block to mount your Google drive account containing PRIME dataset. 
2. Run code blocks one after the other.
3. In code block containing `OCTDataset` uncomment whichever feature you wish to generate and run it for all images. 
Supported features: Canny edge, LaplacianPyramidFusion, Alpha blending, Sobel operator, Canny edge on fusion image level 5, Canny edge on alpha blended image, 7x7 grid - plain - image, and 7x7 grid - canny - image.

## Machine learning models
1. SVM.ipynb
2. KNN.ipynb
3. NaiveBayes.ipynb 
4. UNet-CNN.ipynb

Python notebooks of all machine learning models are written in a modular fashion and are self explanatory with comment, you can just upload them into your colab and start running code blocks to obtain the result. In case of UNet-CNN if you wish to reproduce the best result we have achieved so far, you can upload the `https://drive.google.com/file/d/1lYCOMhJc7ixHgDaTeYXCfGzsPH2qsUF8/view?usp=sharing` file to your colab session and reuse them.

## Presentation
https://drive.google.com/file/d/1MPEJVVdHZ8d3Cn1gICUw8_39dCMwMtUU/view?usp=share_link

   
