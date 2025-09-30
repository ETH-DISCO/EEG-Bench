from .abstract_clinical_task import AbstractClinicalTask
from ...enums.clinical_classes import ClinicalClasses
from ...datasets.clinical import Gruendler2009Dataset
from sklearn.metrics import f1_score
from ...enums.split import Split


class OCDClinicalTask(AbstractClinicalTask):
    def __init__(self):
        super().__init__(
            name="ocd_clinical",
            clinical_classes = [ClinicalClasses.OCD],
            datasets = [
                Gruendler2009Dataset, 
            ],
            subjects_split={
                Gruendler2009Dataset: {
                    Split.TRAIN: list(range(35)),
                    Split.TEST: list(range(35, 46)),
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
