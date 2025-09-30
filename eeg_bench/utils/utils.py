import random
import numpy as np
import torch
import os
import json
from datetime import datetime
from ..config import get_config_value

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_multilabel_tasks():
    return set(["seizure_clinical", "sleep_stages_clinical", "binary_artifact_clinical", "multiclass_artifact_clinical"])

def save_results(
    y_trains,
    y_trues,
    models_names,
    results,
    dataset_names,
    task_name,
):

    # Get the current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    models_names_unique = list(set(models_names))
    models_str = "_".join(models_names_unique) if models_names_unique else "models"

    # Build the filename with task name, models, and timestamp
    filename = os.path.join(get_config_value("results"), "raw", f"{task_name}_{models_str}_{timestamp}.json")

    if task_name in get_multilabel_tasks():
        y_trains = [[[y_2.tolist() for y_2 in y] for y in y_train] for y_train in y_trains]
        y_trues = [[[y_2.tolist() for y_2 in y] for y in y_true] for y_true in y_trues]
    else:
        if isinstance(y_trains[0][0], np.ndarray):
            y_trains = [[y.tolist() for y in y_train] for y_train in y_trains]
            y_trues = [[y.tolist() for y in y_true] for y_true in y_trues]    

    # y_trains = [[y.tolist() if isinstance(y, np.ndarray) else [y_2.tolist() for y_2 in y] for y in y_train] for y_train in y_trains]
    # y_trues = [[y.tolist() if isinstance(y, np.ndarray) else [y_2.tolist() for y_2 in y] for y in y_test] for y_test in y_trues]
    results = [result.tolist() if isinstance(result, np.ndarray) else [y_2.tolist() for y_2 in result] for result in results]

    # Prepare the data to be saved
    data_to_save = {
        "y_train": y_trains,
        "y_test": y_trues,
        "models_names": models_names,
        "results": results,
        "dataset_names": dataset_names,
        "task_name": task_name,
        "timestamp": timestamp
    }

    # Save the results to the file
    json_folder = os.path.join(get_config_value("results"), "raw")
    if not os.path.exists(json_folder):
        os.makedirs(json_folder)
    with open(filename, "w") as f:
        json.dump(data_to_save, f)
    print(f"Results saved to {filename}")
