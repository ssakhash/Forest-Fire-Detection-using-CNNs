# PyroVision: Harnessing ResNet-50 Transfer Learning for Forest Fire Detection

This project implements a Convolutional Neural Network (CNN) based forest fire detection system using PyTorch. I have used a pre-trained ResNet-50 model, fine-tuned it on a custom dataset, and evaluated its performance on detecting forest fires from images.

## Table of Contents

- [Requirements](#requirements)
- [Dataset](#dataset)
- [Usage](#usage)
- [Customization](#customization)
- [License](#license)

## Requirements

To run this project, you need to have the following packages installed:

- Python 3.6 or later
- PyTorch 1.9.0 or later
- torchvision 0.10.0 or later
- NumPy 1.19.5 or later
- tqdm 4.62.3 or later
- matplotlib 3.4.3 or later
- scikit-learn 0.24.2 or later
    
You can install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset should be organized into a folder called 'Dataset' with two subfolders: 'fire' and 'no fire'. These subfolders should contain images of size 224x224 pixels, where the 'fire' folder contains images with fire and the 'no fire' folder contains images without fire.

- Dataset/
  - fire/
    - image1.jpg
    - image2.jpg
    - ...
  - no fire/
    - image1.jpg
    - image2.jpg
    - ...
    
## Usage

After organizing the dataset, you can run the main script to train and test the forest fire detection model:
```bash
python forest_fire_detector.py
```
This script will train the model on 60% of the dataset, validate it on 30% of the dataset, and test it on the remaining 10% of the dataset. The training progress will be shown using a progress bar with the current epoch, loss, and validation accuracy. By default, the script will check for an existing model and ask whether to retrain the model or load the existing one.

## Customization

You can customize the model architecture, training parameters, and dataset split ratios by modifying the forest_fire_detector.py script. For example, you can change the model type, number of classes, learning rate, and batch size as needed. The available command-line arguments are:

- --data-dir: Path to the dataset folder (default: "Dataset")
- --batch-size: Batch size for training (default: 16)
- --learning-rate: Learning rate for the optimizer (default: 0.001)
- --num-epochs: Number of epochs to train (default: 10)
- --model-save-path: Path to save the best model (default: "best_model.pth")
- --retrain: Flag to retrain the model

For example, to train with a different dataset directory, batch size, and learning rate, you can run:
```bash
python forest_fire_detector.py --data-dir /path/to/dataset --batch-size 32 --learning-rate 0.0001
```

## License
This project is licensed under the [MIT License](LICENSE).
