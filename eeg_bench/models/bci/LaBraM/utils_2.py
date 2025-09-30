import numpy as np
from typing import List
import logging

def n_unique_labels(task_name: str) -> int:
    """
    Get the number of unique labels for the given task.
    Args:
        task_name (str): The name of the task.
    Returns:
        int: The number of unique labels.
    """
    if task_name == "Left Hand vs Right Hand MI":
        return 2
    elif task_name == "Right Hand vs Feet MI":
        return 2
    elif task_name == "Left Hand vs Right Hand vs Feet vs Tongue MI":
        return 4
    elif task_name == "Five Fingers MI":
        return 5
    else:
        raise ValueError("Invalid task name: ", task_name)

def map_label(label: str, task_name: str) -> int:
    """
    Map the label to a numerical value.
    Args:
        label (str): The label to map.
        task_name (str): The name of the task.
    Returns:
        int: The mapped numerical value.
    """
    if label is not None:
        if task_name == "Left Hand vs Right Hand MI":
            if label == "left_hand":
                return 0
            elif label == "right_hand":
                return 1
        elif task_name == "Right Hand vs Feet MI":
            if label == "right_hand":
                return 0
            elif label == "feet":
                return 1
        elif task_name == "Left Hand vs Right Hand vs Feet vs Tongue MI":
            if label == "left_hand":
                return 0
            elif label == "right_hand":
                return 1
            elif label == "feet":
                return 2
            elif label == "tongue":
                return 3
        elif task_name == "Five Fingers MI":
            if label == "thumb":
                return 0
            elif label == "index finger":
                return 1
            elif label == "middle finger":
                return 2
            elif label == "ring finger":
                return 3
            elif label == "little finger":
                return 4
        else:
            raise ValueError("Invalid label: ", label)
    else:
        raise ValueError("Label cannot be None")
        
def reverse_map_label(label: int, task_name: str) -> str:
    """
    Reverse map the numerical label to its string representation.
    Args:
        label (int): The numerical label to reverse map.
        task_name (str): The name of the task.
    Returns:
        str: The string representation of the label.
    """
    if task_name == "Left Hand vs Right Hand MI":
        return "left_hand" if label == 0 else "right_hand"
    elif task_name == "Right Hand vs Feet MI":
        return "right_hand" if label == 0 else "feet"
    elif task_name == "Left Hand vs Right Hand vs Feet vs Tongue MI":
        return ["left_hand", "right_hand", "feet", "tongue"][label]
    elif task_name == "Five Fingers MI":
        return ["thumb", "index finger", "middle finger", "ring finger", "little finger"][label]
    else:
        raise ValueError("Invalid label: ", label)

def calc_class_weights(labels: List[np.ndarray], task_name: str) -> List[float]:
    """
    Calculate class weights for the given labels.
    Args:
        labels (List[np.ndarray]): List of numpy arrays containing the labels.
    Returns:
        List[float]: List of weights for each class.
    """
    # Flatten the list of labels
    all_labels = np.concatenate(labels)

    # Map labels to integers
    all_labels = np.array([map_label(label, task_name) for label in all_labels])
    
    # Count the occurrences of each class
    class_counts = np.bincount(all_labels)
    
    # Calculate the total number of samples
    total_samples = len(all_labels)
    
    # Calculate class weights for each class (0 weight if class count is 0)
    n_classes = len(class_counts)
    class_weights = [np.float32(total_samples / (n_classes * count)) if count > 0 else np.float32(0.0) for count in class_counts]
    
    return class_weights
