from .abstract_clinical_task import AbstractClinicalTask
from ...enums.clinical_classes import ClinicalClasses
from ...datasets.clinical import TUEGAbnormalDataset
from sklearn.metrics import f1_score
from ...enums.split import Split
from typing import List, Tuple, Dict
from mne.io import BaseRaw


class AbnormalClinicalTask(AbstractClinicalTask):
    def __init__(self):
        super().__init__(
            name="abnormal_clinical",
            clinical_classes = [ClinicalClasses.ABNORMAL],
            datasets = [
                TUEGAbnormalDataset,
            ],
            subjects_split={
                TUEGAbnormalDataset: {
                    Split.TRAIN: [-1],
                    Split.TEST: [-1],
                },
            },  
        )

    def get_data(
        self, split: Split
    ) -> Tuple[List[List[BaseRaw]], List[List[str]], List[Dict]]:
        """Get the data of the TUEG Abnormal dataset for the given split.
        
        The dataset contains EEG recordings of subjects with normal or abnormal EEGs.
        The subjects are randomly mapped to the corresponding files in the dataset. (Improve me!)

        Args:
            split (Split): The split for which to load the data.
    
        Returns:
            Tuple:
                - List[BaseRaw]: A list of `RawEDF` EEG recordings.
                - List[str]: A list of labels (`"abnormal"` or `"normal"`).
                - Dict: Metadata containing a list of montage types for each EEG signal.
        """
        data = [
            dataset(
                target_class=self.clinical_classes[0],
                subjects=self.subjects_split[dataset][split],
            ).get_data(split)
            for dataset in self.datasets
        ]

        X, y, meta = map(list, zip(*data))
        for m in meta:
            m["task_name"] = self.name
        return X, y, meta

    def get_metrics(self):
        return lambda y, y_pred: f1_score(y, y_pred.ravel())
