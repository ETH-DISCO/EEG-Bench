from .abstract_clinical_task import AbstractClinicalTask
from ...enums.clinical_classes import ClinicalClasses
from ...datasets.clinical import CHBMITDataset
from sklearn.metrics import f1_score
from ...enums.split import Split
from typing import List, Tuple, Dict
from mne.io import Raw


class SeizureClinicalTask(AbstractClinicalTask):
    def __init__(self):
        super().__init__(
            name="seizure_clinical",
            clinical_classes = [ClinicalClasses.SEIZURE],
            datasets = [
                CHBMITDataset,
            ],
            subjects_split={
                CHBMITDataset: {
                    Split.TRAIN: list([ 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23, 24 ]),
                    Split.TEST: list([ 1, 4, 15 ]),
                },
            },
            event_map = {
                "Seizure": 1
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
