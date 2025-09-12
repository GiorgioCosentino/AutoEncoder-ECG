# AutoEncoder-ECG

This repository contains code for implementing an AutoEncoder model using ECG (Electrocardiogram) data. The project is based entirely on Jupyter Notebooks, making it easy to follow, experiment with, and visualize results.

## Overview

An **AutoEncoder** is a type of neural network used to learn efficient codings of unlabeled data. In this project, the AutoEncoder is trained on ECG signals to learn their underlying structure, enabling tasks such as noise reduction, anomaly detection, and feature extraction.

## How the Code Works

1. **Data Loading & Preprocessing**
    - The ECG dataset is loaded, typically in a format such as CSV or NumPy arrays.
    - Data is normalized and optionally segmented to prepare it for the neural network.

2. **Model Architecture**
    - The AutoEncoder consists of two main parts:
        - **Encoder:** Compresses the input ECG signal into a lower-dimensional representation (latent space).
        - **Decoder:** Reconstructs the original ECG signal from the compressed representation.
    - The architecture is implemented using standard deep learning libraries (e.g., TensorFlow or PyTorch) within Jupyter Notebook cells.

3. **Training**
    - The model is trained to minimize the reconstruction error between the original and reconstructed ECG signals.
    - Training parameters such as epochs, batch size, and learning rate can be adjusted within the notebook.

4. **Evaluation & Visualization**
    - After training, the model’s performance is evaluated by visualizing the reconstructed signals and comparing them to the originals.
    - Useful metrics such as Mean Absolute Error (MAE) are calculated.
    - Plots are generated to show the encoding/decoding process and reconstruction quality.

5. **Results**
    - Accuracy: **94.8%**
    - Precision: **99%**
    - Recall: **91.6%**

6. **Applications**
    - The trained AutoEncoder can be used for:
        - Denoising ECG signals.
        - Detecting anomalies (e.g., arrhythmias) by analyzing reconstruction errors.
        - Extracting features for further analysis or classification.

     
## How to Use

1. **Clone the Repository**
    ```bash
    git clone https://github.com/GiorgioCosentino/AutoEncoder-ECG.git
    ```

2. **Open the Jupyter Notebook**
    - Navigate to the directory and launch Jupyter Notebook:
        ```bash
        jupyter notebook
        ```
    - Open the notebook and follow the step-by-step instructions.

3. **Run the Cells**
    - Execute the cells in order to preprocess data, define the AutoEncoder, train the model, and visualize results.

## Requirements

- Python 3.x
- Jupyter Notebook
- NumPy
- pandas
- matplotlib
- TensorFlow or PyTorch (depending on the implementation)

Install the requirements using pip:
```bash
pip install numpy pandas matplotlib tensorflow jupyter
```
or for PyTorch:
```bash
pip install numpy pandas matplotlib torch jupyter
```

## Repository Structure

- `*.ipynb` – Main Jupyter Notebook(s) containing all code, explanations, and results.
- `README.md` – Project overview and instructions.

## License

This project is open-source and available under the [MIT License](LICENSE).
