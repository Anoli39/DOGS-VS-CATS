# Cats vs. Dogs Image Classification

A deep learning project that classifies images of cats and dogs using Convolutional Neural Networks (CNN) and Transfer Learning with TensorFlow/Keras.

##  Project Overview

This project demonstrates a complete workflow of a machine learning project:
1.  Data Preprocessing: Automatically organizing raw dataset.
2.  Baseline Model: Building and training a CNN from scratch.
3.  Advanced Models: Applying Transfer Learning with VGG16 (Feature Extraction & Fine-Tuning).
4.  Comparison: Evaluating and comparing the performance of all three models.

##  Results

| Model | Validation Accuracy | Key Features |
| :--- | :--- | :--- |
| **Baseline CNN** | ~90% | Built from scratch, Data Augmentation |
| **VGG16 (Feature Extraction)** | ~92% | Leveraged pre-trained features |
| VGG16 (Fine-Tuned)| ~98% | Best performance, Adapted pre-trained knowledge |

![Training History]((https://github.com/Anoli39/DOGS-VS-CATS/blob/main/assets/feature_extractor_training_history.png)) *Example of training curves*

## 🛠 Tech Stack

- **Python**
- **TensorFlow / Keras**
- **OpenCV** (for data handling concepts)
- **NumPy, Matplotlib**
- **Google Colab / Jupyter Notebook** (optional)

##  Project Structure

```
dog-vs-cat-classification/
├── src/
│   ├── organize_dataset.py          # Script to sort images into folders
│   ├── cat_dog_cnn.py               # Baseline CNN model
│   ├── train_feature_extractor.py   # Transfer Learning - Feature Extraction
│   ├── train_fine_tuned.py          # Transfer Learning - Fine-Tuning
│   ├── compare_models.py            # Script to compare all models
│   └── predict.py                   # Script to run predictions on new images
├── assets/                          # Contains images for this README
├── .gitignore
└── README.md
```

##  How to Run

1.  **Get the Data**:
    Download the dataset from [Kaggle: Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data) and extract it into the project root.

2.  **Set Up Environment**:
    ```bash
    conda create -n catdog_env python=3.8
    conda activate catdog_env
    pip install -r requirements.txt  # (You can create this with `pip freeze > requirements.txt`)
    ```

3.  **Run the Scripts in Order**:
    ```bash
    cd src
    # 1. Organize the data
    python organize_dataset.py
    # 2. Train the baseline model
    python cat_dog_cnn.py
    # 3. Train the feature extractor model
    python train_feature_extractor.py
    # 4. Fine-tune the model
    python train_fine_tuned.py
    # 5. Compare the models (requires trained .h5 files)
    python compare_models.py
    ```

##  Author
Anoli39

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
