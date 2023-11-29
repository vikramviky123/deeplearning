import os
from pathlib import Path
import zipfile
import requests
import numpy as np

from sklearn.metrics import roc_curve


def splits(dataset, TRAIN_RATIO, VAL_RATIO):
    DATASET_SIZE = len(dataset)

    train_dataset = dataset.take(int(TRAIN_RATIO*DATASET_SIZE))

    val_test_dataset = dataset.skip(int(TRAIN_RATIO*DATASET_SIZE))
    val_dataset = val_test_dataset.take(int(VAL_RATIO*DATASET_SIZE))

    return train_dataset, val_dataset


def download_zip_file(url: str, save_path: Path):
    response = requests.get(url)

    fileDir, fileName = os.path.split(save_path)

    os.makedirs(fileDir, exist_ok=True)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"Download successful. File saved at {save_path}")
    else:
        print(f"Error {response.status_code}: Unable to download the file.")


def extract_zip_file(zip_file_path, extract_path):

    fileDir, fileName = os.path.split(extract_path)
    os.makedirs(fileDir, exist_ok=True)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Extraction successful. Files extracted to {extract_path}")


def labels_with_thresh(pred_prob, optimal_threshold):
    pred_labels = []
    for i in pred_prob:
        if i > optimal_threshold:
            pred_labels.append(1)
        else:
            pred_labels.append(0)
    return pred_labels

# thresholds --> for each thresholds compute fpr and tpr


def optimum_thresh(y_labels, pred_prob):
    fpr, tpr, thresholds = roc_curve(y_labels, pred_prob)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold, fpr, tpr
