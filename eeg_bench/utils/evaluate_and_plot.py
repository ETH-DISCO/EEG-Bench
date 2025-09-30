from ..config import get_config_value
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    cohen_kappa_score,
)
from .utils import get_multilabel_tasks

def one_hot_encode(y):
        encoder = OneHotEncoder(sparse_output=False)
        y_reshaped = np.array(y).reshape(-1, 1)
        return encoder.fit_transform(y_reshaped)

from datetime import datetime

def print_classification_results(
        y_trains, y_trues, model_names, y_preds, dataset_names, task_name, metrics):
    # Assuming y_train and y_test are lists of numpy arrays

    # Gather basic statistics
    train_samples = [len(y) for y in y_trains[0]]
    test_samples = [len(y) for y in y_trues[0]]

    def class_distribution(y_list):
        encoder = LabelEncoder()
        distributions = []
        for y in y_list:
            if task_name in get_multilabel_tasks():
                # multilabel case: y is list of multilabels (1 multilabel = 1D numpy array)
                encoded_y = np.concatenate(y)
            else:
                encoded_y = encoder.fit_transform(y)  # Encode string labels to integers
            distributions.append(np.bincount(encoded_y).tolist())
        return distributions

    train_distributions = class_distribution(y_trains[0])
    test_distributions = class_distribution(y_trues[0])

    # Create a DataFrame for better formatting
    task_data = {
        "Dataset": [f"{dataset_names[i]}" for i in range(len(dataset_names))],
        "Train Samples": train_samples,
        "Test Samples": test_samples,
        "Train Class Distribution": [str(dist) for dist in train_distributions],
        "Test Class Distribution": [str(dist) for dist in test_distributions],
    }
    task_table = pd.DataFrame(task_data)

    output = ""
    output += "\n" + "=" * 24 + " Task Overview " + "=" * 24 + "\n"
    output += f"{'Task: ' + task_name:^60}\n"
    output += task_table.to_string(index=False)
    output += "\n"

    # Function to calculate metrics
    def calculate_metrics(y_true, y_pred, no_encoder=False):
        encoder = LabelEncoder()
        if not no_encoder:
            y_true = encoder.fit_transform(y_true)
            y_pred = encoder.transform(y_pred)
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
            "Weighted F1": f1_score(y_true, y_pred, average="weighted"),
            "Macro F1": f1_score(y_true, y_pred, average="macro"),
            "ROC AUC": (
            roc_auc_score(one_hot_encode(y_true), one_hot_encode(y_pred), multi_class="ovr")
            if len(np.unique(y_true)) > 2
            else (roc_auc_score(y_true, y_pred) if len(y_true) > 0 else 0)
            ),
            "Average Precision": (
            average_precision_score(y_true, y_pred)
            if len(np.unique(y_true)) == 2
            else average_precision_score(one_hot_encode(y_true), one_hot_encode(y_pred), average="macro")
            ),
            "Cohen Kappa": cohen_kappa_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average="weighted"),
            "Recall": recall_score(y_true, y_pred, average="weighted"),
        }

    # Iterate over models and create tables
    for model_name, y_pred, y_test in zip(model_names, y_preds, y_trues):
        output += "-" * 25 + f" {model_name} " + "-" * 25 + "\n"

        # Overall metrics
        if task_name in get_multilabel_tasks():
            # multilabel case: flatten the 2D [#samples, #multilabels] to 1D [#samples * #multilabels]
            combined_y_test = np.concatenate([np.concatenate(y) for y in y_test])
            combined_y_pred = np.concatenate([np.concatenate(y) for y in y_pred])
            combined_metrics = calculate_metrics(combined_y_test, combined_y_pred, no_encoder=True)
        else:
            combined_y_test = np.concatenate(y_test)
            combined_y_pred = np.concatenate(y_pred)
            combined_metrics = calculate_metrics(combined_y_test, combined_y_pred)

        # Create a table for combined and per-dataset metrics
        results = []

        # Add overall metrics to the table
        results.append([task_name] + list(combined_metrics.values()))
        if task_name == "parkinsons_clinical":
            singh_index = dataset_names.index("Singh2020") if "Singh2020" in dataset_names else None
            if singh_index is not None:
                singh_metrics = calculate_metrics(y_test[singh_index], y_pred[singh_index])
                results.append(["Held-Out"] + list(singh_metrics.values()))
        elif task_name == "Left Hand vs Right Hand MI":
            zhou_index = dataset_names.index("Zhou2016") if "Zhou2016" in dataset_names else None
            if zhou_index is not None:
                zhou_metrics = calculate_metrics(y_test[zhou_index], y_pred[zhou_index])
                results.append(["Held-Out"] + list(zhou_metrics.values()))
        elif task_name == "Right Hand vs Feet MI":
            zhou_index = dataset_names.index("Zhou2016") if "Zhou2016" in dataset_names else None
            if zhou_index is not None:
                zhou_metrics = calculate_metrics(y_test[zhou_index], y_pred[zhou_index])
                results.append(["Held-Out"] + list(zhou_metrics.values()))
        
        # Add per-dataset metrics
        for i, (this_y_test, this_y_pred) in enumerate(zip(y_test, y_pred)):
            if task_name in get_multilabel_tasks():
                this_y_test = np.concatenate(this_y_test)
                this_y_pred = np.concatenate(this_y_pred)
                dataset_metrics = calculate_metrics(this_y_test, this_y_pred, no_encoder=True)
            else:
                dataset_metrics = calculate_metrics(this_y_test, this_y_pred)
            results.append([f"{dataset_names[i]}"] + list(dataset_metrics.values()))

        # Create a DataFrame for tabular formatting
        metrics_table = pd.DataFrame(
            results,
            columns=["", "Accuracy", "Balanced Accuracy", "Weighted F1", "Macro F1", "ROC AUC", "Average Precision", "Cohen Kappa", "Precision", "Recall"],
        )

        # Append table to output
        output += metrics_table.to_string(index=False)
        output += "\n"

    # Print the results to the console
    print(output)

    # Save results to a file if specified
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(get_config_value("results"), f"classification_results_{task_name}_{timestamp}.txt")
    with open(filename, 'w') as f:
        f.write(output)
    print(f"Results saved to {filename}")


def save_class_distribution_plots(dataset_names, train_distribution, test_distribution, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    for i, dataset in enumerate(dataset_names):
        plt.figure(figsize=(8, 4))
        plt.bar(range(len(train_distribution[i])), train_distribution[i], alpha=0.7, label="Train")
        plt.bar(range(len(test_distribution[i])), test_distribution[i], alpha=0.7, label="Test")
        plt.xlabel("Class Label")
        plt.ylabel("Count")
        plt.title(f"Class Distribution - {dataset}")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"class_distribution_{dataset}.png"))
        plt.close()

def save_evaluation_plots(model_names, dataset_names, all_metrics, task_name, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name, metrics_table in zip(model_names, all_metrics):
        plt.figure(figsize=(10, 6))
        metrics_table.set_index("Dataset").plot(kind="bar", figsize=(10, 6))
        plt.title(f"Evaluation Metrics - {model_name} - {task_name}")
        plt.ylabel("Score")
        plt.xticks(rotation=45)
        plt.legend(loc="lower right")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"metrics_{model_name}_{task_name}.png"))
        plt.close()

def generate_classification_plots(y_trains, y_trues, model_names, y_preds, dataset_names, task_name, metrics, output_dir="plots"):
    def calculate_metrics(y_true, y_pred, no_encoder=False):
        encoder = LabelEncoder()
        if not no_encoder:
            y_true = encoder.fit_transform(y_true)
            y_pred = encoder.transform(y_pred)
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average="weighted"),
            "Recall": recall_score(y_true, y_pred, average="weighted"),
            "F1 Score": f1_score(y_true, y_pred, average="weighted"),
            "AUC": (
                roc_auc_score(one_hot_encode(y_true), one_hot_encode(y_pred), multi_class="ovr")
                if len(np.unique(y_true)) > 2
                else roc_auc_score(y_true, y_pred)
            ),
        }
    
    all_metrics = []
    
    for model_name, y_pred, y_test in zip(model_names, y_preds, y_trues):
        if task_name in get_multilabel_tasks():
            # multilabel case: flatten the 2D [#samples, #multilabels] to 1D [#samples * #multilabels]
            combined_y_test = np.concatenate([np.concatenate(y) for y in y_test])
            combined_y_pred = np.concatenate([np.concatenate(y) for y in y_pred])
            combined_metrics = calculate_metrics(combined_y_test, combined_y_pred, no_encoder=True)
        else:
            combined_y_test = np.concatenate(y_test)
            combined_y_pred = np.concatenate(y_pred)
            combined_metrics = calculate_metrics(combined_y_test, combined_y_pred)
        
        results = [[task_name] + list(combined_metrics.values())]
        for i, (this_y_test, this_y_pred) in enumerate(zip(y_test, y_pred)):
            if task_name in get_multilabel_tasks():
                this_y_test = np.concatenate(this_y_test)
                this_y_pred = np.concatenate(this_y_pred)
                dataset_metrics = calculate_metrics(this_y_test, this_y_pred, no_encoder=True)
            else:    
                dataset_metrics = calculate_metrics(this_y_test, this_y_pred)
            results.append([f"{dataset_names[i]}"] + list(dataset_metrics.values()))
        
        metrics_table = pd.DataFrame(
            results,
            columns=["Dataset", "Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1 Score", "AUC"],
        )
        all_metrics.append(metrics_table)
    
    # Save evaluation metric plots
    save_evaluation_plots(model_names, dataset_names, all_metrics, task_name, output_dir)
