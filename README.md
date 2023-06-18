# Gender Detection using SIFT Feature Extraction and SVM/XGBoost Models

This project aims to develop two machine learning models that can detect the gender of a person from their facial features. The models use Scale-Invariant Feature Transform (SIFT) features to extract information from facial images and classify them as either male or female. The models are implemented using Support Vector Machines (SVM) and XGBoost algorithms.

## Getting Started

To get started with this project, you will need to have Python 3 installed on your computer. You will also need to install the following Python libraries:

- numpy
- pandas
- scikit-learn
- opencv-python
- xgboost

You can install these libraries using pip, as follows:

```
pip install numpy
pip install pandas
pip install scikit-learn
pip install opencv-python
pip install xgboost
```

Once you have installed the required libraries, you can run the gender detection models by running the `svm_gender_detection.py` and `xgboost_gender_detection.py` files. The models will prompt you to enter the path to an image of a face, and they will then classify the image as either male or female.

## Project Structure

The project is structured as follows:

- `svm_gender_detection.py` - This is the script that runs the gender detection model using SIFT features and SVM classifier.
- `xgboost_gender_detection.py` - This is the script that runs the gender detection model using SIFT features and XGBoost classifier.
- `data` - This folder contains a sample dataset of facial images.
- `data_processing.py` - This module contains functions for processing and cleaning the facial image data.
- `feature_extraction.py` - This module contains functions for extracting SIFT features from the facial image data.
- `model_training.py` - This module contains functions for training and evaluating the machine learning models.

## Contributing

Contributions to this project are welcome. If you would like to contribute, please fork the project and submit a pull request with your changes.
