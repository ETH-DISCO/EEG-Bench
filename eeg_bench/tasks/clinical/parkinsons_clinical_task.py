from .abstract_clinical_task import AbstractClinicalTask
from ...enums.clinical_classes import ClinicalClasses
from ...datasets.clinical import (
    Cavanagh2018aDataset,
    Cavanagh2018bDataset,
    Singh2018Dataset,
    Brown2020Dataset,
    Singh2020Dataset,
    Singh2021Dataset,
)
from sklearn.metrics import f1_score
from ...enums.split import Split


class ParkinsonsClinicalTask(AbstractClinicalTask):
    def __init__(self):
        super().__init__(
            name="parkinsons_clinical",
            clinical_classes = [ClinicalClasses.PARKINSONS],
            datasets = [
                Cavanagh2018aDataset,
                Cavanagh2018bDataset,
                Singh2018Dataset,
                Brown2020Dataset,
                Singh2020Dataset,
                Singh2021Dataset,
            ],
            subjects_split={
                Cavanagh2018aDataset: {
                    Split.TRAIN: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 25, 26, 28, 29, 30, 31, 34, 35, 38, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51],
                    Split.TEST: [17, 18, 19, 20, 21, 22, 23, 24, 27, 32, 33, 36, 37, 39, 42, 52],
                },
                Cavanagh2018bDataset: {
                    Split.TRAIN: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 28, 29, 31, 32, 33, 34, 37, 38, 41, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54],
                    Split.TEST: [20, 21, 22, 23, 24, 25, 26, 27, 30, 35, 36, 39, 40, 42, 45, 55],
                },
                Singh2018Dataset: {
                    Split.TRAIN: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 28, 29, 31, 32, 34, 37, 38, 41, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54],
                    Split.TEST: [20, 21, 22, 23, 24, 25, 26, 27, 30, 35, 36, 39, 40, 42, 45, 55],
                },
                Brown2020Dataset: {
                    Split.TRAIN: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 28, 29, 31, 32, 33, 34, 37, 38, 41, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54],
                    Split.TEST: [20, 21, 22, 23, 24, 25, 26, 27, 30, 35, 36, 39, 40, 42, 45, 55],
                },
                Singh2020Dataset: {
                    Split.TRAIN: [],
                    Split.TEST: list(range(39)),
                },
                Singh2021Dataset: {
                    Split.TRAIN: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87],
                    Split.TEST: [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                }
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
