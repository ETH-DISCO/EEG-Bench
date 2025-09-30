from .abstract_clinical_task import AbstractClinicalTask
from ...enums.clinical_classes import ClinicalClasses
from ...datasets.clinical import SleepTelemetryDataset
from sklearn.metrics import f1_score
from ...enums.split import Split
from typing import List, Tuple, Dict
from mne.io import Raw


class SleepStagesClinicalTask(AbstractClinicalTask):
    def __init__(self):
        super().__init__(
            name="sleep_stages_clinical",
            clinical_classes = [ClinicalClasses.WAKE, ClinicalClasses.N1, ClinicalClasses.N2, ClinicalClasses.N34, ClinicalClasses.REM],
            datasets = [
                SleepTelemetryDataset,
            ],
            subjects_split={
                SleepTelemetryDataset: {
                    Split.TRAIN: list([2, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 24]),
                    Split.TEST: list([1, 4, 5, 6, 16]),
                },
            },
            event_map = {
                "Sleep stage W": 1,
                "Sleep stage 1": 2,
                "Sleep stage 2": 3,
                "Sleep stage 3/4": 4,
                "Sleep stage R": 5
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
