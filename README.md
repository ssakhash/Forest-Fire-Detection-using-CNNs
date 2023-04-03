# Forest Fire Detection using CNNs

This project implements a Convolutional Neural Network (CNN) based forest fire detection system using PyTorch. We use a pre-trained ResNet-50 model, fine-tune it on a custom dataset, and evaluate its performance on detecting forest fires from images.

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
This script will train the model on 70% of the dataset, validate it on 20% of the dataset, and test it on the remaining 10% of the dataset. The training progress will be shown using a progress bar with the current epoch, loss, and validation accuracy.

## Customization

You can customize the model architecture, training parameters, and dataset split ratios by modifying the ForestFireDetector class instantiation and its method calls in the forest_fire_detector.py script. For example, you can change the model type, number of classes, learning rate, and batch size as needed.

## License
This project is licensed under the [MIT License](LICENSE).
