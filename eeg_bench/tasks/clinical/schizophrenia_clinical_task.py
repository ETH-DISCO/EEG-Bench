from .abstract_clinical_task import AbstractClinicalTask
from ...enums.clinical_classes import ClinicalClasses
from ...datasets.clinical import Albrecht2019Dataset
from sklearn.metrics import f1_score
from ...enums.split import Split


class SchizophreniaClinicalTask(AbstractClinicalTask):
    def __init__(self):
        super().__init__(
            name="schizophrenia_clinical",
            clinical_classes = [ClinicalClasses.SCHIZOPHRENIA],
            datasets = [
                Albrecht2019Dataset, 
            ],
            subjects_split={
                Albrecht2019Dataset: {
                    Split.TRAIN: list(range(63)),
                    Split.TEST: list(range(63, 76)),
                },
            }
        )
        
    def get_data(
        self, split: Split
    ):
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
