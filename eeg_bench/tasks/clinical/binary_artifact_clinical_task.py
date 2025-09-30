from .abstract_clinical_task import AbstractClinicalTask
from ...enums.clinical_classes import ClinicalClasses
from ...datasets.clinical import TUARDataset
from sklearn.metrics import f1_score
from ...enums.split import Split
from typing import List, Tuple, Dict
from mne.io import Raw


class ArtifactBinaryClinicalTask(AbstractClinicalTask):
    def __init__(self):
        super().__init__(
            name="binary_artifact_clinical",
            clinical_classes = [ClinicalClasses.ARTIFACT],
            datasets = [
                TUARDataset,
            ],
            subjects_split={
                TUARDataset: {
                    Split.TRAIN: list([0, 2, 3, 7, 8, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 25, 28, 31, 33, 34, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 53, 54, 55, 56, 58, 59, 60, 61, 62, 63, 64, 65, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 95, 97, 98, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 132, 133, 134, 135, 137, 138, 139, 140, 142, 143, 145, 146, 147, 149, 150, 151, 152, 154, 155, 156, 157, 158, 160, 161, 162, 164, 165, 166, 167, 168, 169, 170, 171, 174, 175, 178, 179, 181, 183, 185, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 200, 202, 203, 204, 205, 206, 209]),
                    Split.TEST: list([1, 131, 4, 5, 6, 136, 9, 10, 141, 144, 17, 148, 22, 23, 24, 153, 26, 27, 29, 30, 159, 32, 163, 35, 39, 172, 173, 47, 176, 177, 180, 182, 184, 57, 186, 66, 199, 201, 78, 207, 208, 210, 211, 212, 89, 96, 99, 112 ]),
                },
            },
            event_map = {
                "Eye Movement": 1,
                "Muscle Artifact": 1,
                "Electrode Artifact": 1,
                "Chewing": 1,
                "Shivers": 1,
                "Electrode Pop": 1,
            },
            chunk_len_s = 16,
            num_labels_per_chunk = 16
        )

    def get_data(
        self, split: Split
    ) -> Tuple[List[List[List[Raw]]], List[List[str]], List[Dict]]:
        return super().get_data(split)

    def get_metrics(self):
        return lambda y, y_pred: f1_score(y, y_pred.ravel())
