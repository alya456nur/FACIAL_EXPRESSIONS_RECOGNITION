FACIAL EXPRESSION RECOGNITION USING CNN
=======================================

ABOUT THE PROJECT
-----------------

Facial Expression Recognition is a Python-based machine learning application that uses a Convolutional Neural Network (CNN) to recognize human facial expressions.

The system analyzes facial images and predicts five different emotions: Angry, Happy, Neutral, Sad, and Surprise.

It provides two ways to recognize facial expressions:
- Real-time facial expression detection using a webcam.
- Facial expression prediction using uploaded images.

The application features a graphical user interface (GUI) developed with PyQt5, allowing users to interact with the system easily.


FEATURES
--------

1. Real-Time Emotion Detection
   - Detect faces through a webcam.
   - Identify facial expressions in real time.
   - Display the predicted emotion and confidence percentage.

2. Image-Based Emotion Recognition
   - Upload images from the device.
   - Analyze facial expressions in uploaded images.
   - Display the predicted emotion and confidence score.

3. CNN Model Training
   - Train a Convolutional Neural Network (CNN).
   - Apply image preprocessing and data augmentation.
   - Use class weights to help address class imbalance.
   - Save the trained model for future predictions.

4. Model Performance Evaluation
   - Generate a classification report.
   - Calculate accuracy, precision, recall, and F1-score.
   - Display a confusion matrix.

5. Graphical User Interface
   - Interactive desktop application.
   - Webcam detection and image prediction options.
   - Display prediction results through the application interface.


RECOGNIZED EMOTIONS
-------------------

The system recognizes five facial expressions:

1. Angry
2. Happy
3. Neutral
4. Sad
5. Surprise


TECHNOLOGIES USED
-----------------

- Python - Main programming language.
- TensorFlow - Machine learning framework.
- Keras - CNN model development and training.
- OpenCV - Face detection and image processing.
- PyQt5 - Graphical User Interface (GUI).
- NumPy - Numerical operations.
- Scikit-learn - Model evaluation and performance metrics.
- Matplotlib - Data visualization.
- Seaborn - Confusion matrix visualization.

GETTING STARTED
---------------

PREREQUISITES

Make sure you have installed:

- Python 3.x
- pip (Python package manager)
- Webcam (for real-time emotion detection)


INSTALLATION

1. Clone the repository:

   git clone https://github.com/alya456nur/FACIAL_EXPRESSIONS_RECOGNITION.git

2. Navigate to the project directory:

   cd facial-expression-recognition

3. Install the required dependencies:

   pip install tensorflow opencv-python numpy pillow PyQt5 scikit-learn matplotlib seaborn


RUNNING THE APPLICATION
-----------------------

To launch the facial expression recognition application:

   python testUI.py

The application provides the following options:

- Detect Emotion (Webcam)
  Activate the webcam to detect and recognize facial expressions
  in real time.

- Predict from Image
  Select images from your device to predict their facial
  expressions.

- Exit
  Close the application.


MODEL TRAINING
--------------

To train the CNN model using the training dataset:

   python train_model.py

The trained model will be saved in the models directory as:

   models/emotion_model1.keras

Note: Model training requires the dataset to be available in
the expected directory structure.


MODEL EVALUATION
----------------

To evaluate the trained model using the test dataset:

   python performance.py

The evaluation script generates:

- Classification Report
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix


SCREENSHOTS
-----------

Application screenshots will be added here.

- Main Application Interface
- Real-Time Webcam Detection
- Image-Based Emotion Prediction
- Prediction Results


LICENSE
-------

This project was developed for educational purposes.

License and redistribution permissions have not yet been specified.


DEVELOPER
---------

Developed using Python, TensorFlow, Keras, OpenCV, and PyQt5.
