# DIT5411-LeeTszChunJamie
Due to the limitation of the Github file size, full version will be on One Drive
Link: https://vtcmca-my.sharepoint.com/:f:/g/personal/250586895_stu_vtc_edu_hk/IgB-BLr2y4mTS5gMt_aAJv1oAdpQW4O1njb-a7yvTQyehvY?e=stavF1

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

**Data Argumentation**
Since the original dataset contains a limited number of samples for each character, image augmentation was applied to increase training data diversity.
During data augmentation, a random augmentation strategy was adopted. For each generated training image, one or two augmentation techniques were randomly selected from rotation (±15°), scaling (90%-110%), translation (±5 pixels), contrast adjustment, and Gaussian noise injection. This approach increases training diversity while preserving the original characteristics of handwritten Chinese characters.

Augmentation Techniques
_1. Rotation_
Random image rotations within ±15 degrees.
Purpose:
Simulate different handwriting angles.
Improve robustness against tilted writing.

_2. Scaling_
Random zooming between 90% and 110%.
Purpose:
Handle different writing sizes.

_3. Translation_
Random horizontal and vertical movement.
Purpose:
Simulate various writing positions.

_4. Contrast Adjustment_
Random contrast modification.
Purpose:
Improve performance under different scanning conditions.

_5. Noise Injection_
Small Gaussian noise added to images.
Purpose:
Reduce overfitting.
Improve generalization.

Using augmentation, each character class was expanded to approximately 200 training samples.

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
==================
Test Accuracy
==================
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
