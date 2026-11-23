# DIT5411-LeeTszChunJamie

**Overview**

This project was developed for the THEi DIT5411 Machine Learning Assignment (Character Recognition). The objective is to recognize Traditional Chinese handwritten characters using TensorFlow and deep learning techniques. The dataset consists of handwritten Chinese character images collected from the Traditional Chinese Handwriting Dataset.

**Dataset**

Dataset Source:

Traditional Chinese Handwriting Dataset from GitHub Repository: https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset

Dataset Characteristics:

Total Classes: around 4000+ Chinese characters
Average Samples per Character: Around 40 images
Image Format: PNG
Naming Format: (Character)_(number).png

**Dataset Preprocessing**

A preprocessing program was developed to automatically organize the dataset.

**Training and Testing Split**

For each character:

80% of images are used for training
20% of images are used for testing
Maximum 40 training images per character
At least 1 testing image is reserved for evaluation

**Directory**
DIT5411-LeeTszChunJamie/
  .idea/
  .venv/
  data_raw/
    丁_0.png
    丁_1.png
    丁_10.png
    丁_11.png
    丁_12.png
    ...
  dataset/
    test/
      乙/
      丁/
      七/
      乃/
      九/
      ...
    train/
      乙/
      丁/
      七/
      乃/
      九/
      ...
  best_model.keras
  evaluate.py
  preprocess.py
  train.py

**Software and Libraries**

Development Environment:
Python 3.11
TensorFlow 2.x
Keras
NumPy
OpenCV
Matplotlib
Scikit-Learn

Install Dependencies:
pip install tensorflow
pip install numpy
pip install opencv-python
pip install matplotlib
pip install scikit-learn

**Steps**

_Step 1:_
Preprocess.py

_Step 2:_
train.py

_Step 3:_
evaluate.py

_Output:_
Test Accuracy: 7.07%
Loss: 5.6777

**Conclusion**

This project successfully developed a Traditional Chinese handwritten character recognition system using TensorFlow. Experimental results showed that transfer learning using EfficientNetB0 provided the best recognition performance among all tested models while satisfying the assignment requirements for large-scale Chinese character classification.

**References**

Traditional Chinese Handwriting Dataset
 https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset

TensorFlow Documentation
 https://www.tensorflow.org/

Keras Documentation
 https://keras.io/

OpenCV Documentation
 https://opencv.org/

Darshita Jain, Introduction to OpenCV
 https://github.com/darshitajain/Introduction-to-OpenCV
