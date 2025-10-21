Paddy Disease Detection Using CNN

Project Description:

A CNN-based model to detect and classify diseases in paddy leaves. The model can classify images into 10 categories: 9 diseases and 1 healthy class.

Class Names:

1. Bacterial_leaf_blight

2. Bacterial_leaf_streak

3. Bacterial_panicle_blight

4. Blast

5. Brown_spot

6. Dead_heart

7. Downy_mildew

8. Hispa

9. Normal

10. Tungro

Dataset:

Images of paddy leaves organized into folders corresponding to the class names from kaggle

Requirements:

Python 3.x

TensorFlow / Keras

NumPy

OpenCV / Pillow

Matplotlib / Seaborn

Usage

Clone the repository:

git clone <repository-url>
cd paddy-disease-detection


Install dependencies:

pip install -r requirements.txt


Run training:

python train_model.py


Evaluate the model:

python evaluate_model.py


Predict on new images:

python predict.py --image path_to_image
