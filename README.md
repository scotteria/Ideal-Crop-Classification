# Crop Disease Detections Using Machine Learning/ Tensorflow
Author: Scotteria Scott 
Program : Morehouse Machine Learning Technichian Postbacc Program 


# Overview 
Many studies show that by the year 2050 there will not be enough food to feed the population. This model attempts to  solve this problem by determining whether crop leaf disease can be identified early in crops from images using machine learning and tensor flow. The stakeholders are farmers and production companies that use organic products.By developing an automated image classification system farmers and researchers could detect disease and take preventative action faster which can reduce crop loss and improve food security. During this project I will use a dataset from Kaggle  Ecosystem Plant Village Disease in all import plant crops. I believe a good outcome would be this project producing a reproducible pipeline with clear results so it can be used in real time with farmers.

# Accessing the Data
import kagglehub
import os

# Download latest version
path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")

print("Path to dataset files:", path)

# Methods 
Dataset: PlantVillage (Kaggle, 38 classes of healthy and diseased leaves).


Preprocessing: Images resized to 224×224, normalized, and augmented (flip, rotation, zoom, contrast).


Model: MobileNetV2 (pretrained on ImageNet) with added dense layers (ReLU + Dropout + Softmax).


Training:


Phase 1: froze base network, trained classifier head.


Phase 2: fine-tuned top MobileNetV2 layers (low LR).


Optimizer = Adam, Loss = Sparse Categorical Cross-Entropy.


Class weights used to balance categories.


Evaluation: Validation accuracy, precision/recall/F1, and confusion matrix.


Deployment: Saved in TensorFlow and TFLite formats.


# References 


Barbedo, J. G. A. (2018). Impact of dataset size and variety on the effectiveness of deep learning and transfer learning for plant disease classification. Computers and Electronics in Agriculture, 153, 46–53. https://doi.org/10.1016/j.compag.2018.08.013


Ferentinos, K. P. (2018). Deep learning models for plant disease detection and diagnosis. Computers and Electronics in Agriculture, 145, 311–318. https://doi.org/10.1016/j.compag.2018.01.009


Food and Agriculture Organization of the United Nations (FAO). (2019). The state of food and agriculture: Moving forward on food loss and waste reduction. http://www.fao.org/3/ca6030en/ca6030en.pdf


NIAID Data Ecosystem (2024). Crop Health Monitoring Dataset: Disease Classification in Maize, Tea, Tomatoes, Apples, and Beans. https://data.niaid.nih.gov/resources?id=zenodo_10628734


Kaggle/Plantvilliage Disease Detection 
