# Federated and Split Learning for Plant Disease Classification

This repository contains implementations of Federated Learning (FL), Split Learning (SL), and Split Federated Learning (SFL) using MobileNetV2 for plant disease classification on the PlantVillage dataset. The project explores different distributed learning paradigms and incorporates various privacy-preserving mechanisms.

## Table of Contents

-   [Project Title](#federated-and-split-learning-for-plant-disease-classification)
-   [Description](#description)
-   [Features](#features)
-   [Dataset](#dataset)
-   [Installation](#installation)
-   [Usage](#usage)
-   [Privacy Evaluation](#privacy-evaluation)
-   [File Structure](#file-structure)
-   [Contributing](#contributing)
-   [License](#license)

## Description

This project aims to demonstrate and evaluate different distributed machine learning approaches (Federated Learning, Split Learning, and Split Federated Learning) for image classification tasks, specifically plant disease detection. It utilizes the MobileNetV2 architecture and the PlantVillage dataset. A key aspect of this work is the integration and analysis of privacy-preserving techniques within these distributed learning frameworks.

## Features

*   **Federated Learning (FL)**: Implementation of a standard federated learning setup where clients train local models and send updates to a central server for aggregation.
*   **Split Learning (SL)**: Implementation of split learning where the model is split between the client and server, with clients performing initial layers and servers completing the classification.
*   **Split Federated Learning (SFL)**: A hybrid approach combining aspects of both federated and split learning.
*   **MobileNetV2 Architecture**: Utilizes the MobileNetV2 convolutional neural network for efficient image classification.
*   **PlantVillage Dataset**: Designed for plant disease classification, supporting various plant species and disease types.
*   **Privacy Mechanisms**:
    *   **Differential Privacy (DP)**: Implemented through gradient clipping and Gaussian noise addition (in FL and SFL).
    *   **Activation Defenses**: Clipping, L2 norm clipping, Gaussian noise, and quantization applied to activations at the cut layer (in SL and SFL).
    *   **Gradient Defenses**: L2 norm clipping and Gaussian noise applied to gradients (in SL and SFL).
*   **Privacy Evaluation**: Tools for assessing the privacy leakage using metrics like Membership Inference Attacks (MIA) and Activation Inversion.

## Dataset

The project uses the **PlantVillage Dataset** for plant disease classification.

**Download Instructions:**
The dataset can be downloaded from Kaggle: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)

**Structure:**
The dataset should be organized in a directory structure accessible by the scripts. The `setup_dataset` function in the scripts attempts to locate the dataset in common paths like `./dataset/plantvillage`, `./plantvillage_dataset`, or `./PlantVillage`. It also supports pre-split `train` and `val` (or `test`) subdirectories, or can perform an 80/20 stratified split if a single directory is provided.

Example expected structure:

```
./dataset/plantvillage/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── class2/
│       ├── img3.jpg
│       └── img4.jpg
└── val/
    ├── class1/
    │   ├── img5.jpg
    │   └── img6.jpg
    └── class2/
        ├── img7.jpg
        └── img8.jpg
```

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your_username/your_repository_name.git
    cd your_repository_name/SplitFed
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install dependencies:**

    ```bash
    pip install torch torchvision scikit-learn pandas matplotlib pillow numpy
    ```

    *Note: Ensure you have a compatible version of `torch` for your CUDA setup if you plan to use GPU acceleration.*

## Usage

The repository contains several Python scripts, each implementing a different distributed learning scenario or privacy evaluation.

### Training Scripts

To run the training simulations, navigate to the `SplitFed` directory and execute the desired script.

*   **Federated Learning (FL)**:
    ```bash
    python FL_MobileNetV2_PlantVillage.py
    ```

*   **Split Learning (SL)**:
    ```bash
    python SL_MobileNetV2_PlantVillage.py
    ```

*   **Split Federated Learning (SFL)**:
    ```bash
    python SFLV1_MobileNetV2_PlantVillage.py
    ```

Each script will output training and testing progress to the console.

### Configuration

Parameters such as `num_users`, `epochs`, `learning_rate`, and privacy defense settings (`DP_CLIP_NORM`, `DP_NOISE_STD`, `ACT_CLIP_VALUE`, etc.) can be adjusted directly within each Python script.

## Privacy Evaluation

The `privacy_eval.py` script provides tools to evaluate the privacy leakage of the trained models.

**Usage:**

```bash
python privacy_eval.py --dataset <path_to_plantvillage_dataset> \
                       --check mia activation_inversion \
                       --methods FL SL SFL \
                       --samples 16
```

*   `--dataset`: Path to your PlantVillage dataset.
*   `--check`: Specify privacy metrics to evaluate (e.g., `mia` for Membership Inference Attack, `activation_inversion` for Activation Inversion).
*   `--methods`: Specify which distributed learning methods to evaluate (e.g., `FL`, `SL`, `SFL`).
*   `--samples`: Number of samples to use for certain evaluations (e.g., activation inversion).

The script will print metrics such as MIA AUC/attack accuracy and PSNR for activation inversion, indicating the level of privacy.

## File Structure

```
SplitFed/
├── .gitignore
├── FL_MobileNetV2_PlantVillage.py    # Federated Learning implementation
├── SFLV1_MobileNetV2_PlantVillage.py # Split Federated Learning implementation
├── SL_MobileNetV2_PlantVillage.py    # Split Learning implementation
├── federated_learning.py             # Core federated learning utilities (might be a base or older version)
└── privacy_eval.py                   # Script for evaluating privacy metrics
```

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License.
