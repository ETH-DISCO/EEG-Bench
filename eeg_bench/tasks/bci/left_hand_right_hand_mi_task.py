from .abstract_bci_task import AbstractBCITask
from ...datasets.bci import (
    BCICompIV2aMDataset,
    BCICompIV2bMDataset,
    Weibo2014MDataset,
    Cho2017MDataset,
    Liu2022MDataset,
    Schirrmeister2017MDataset,
    PhysionetMIMDataset,
    Zhou2016MDataset,
    Kaya2018Dataset,
)
from ...enums.bci_classes import BCIClasses
from sklearn.metrics import f1_score
from ...enums.split import Split


class LeftHandvRightHandMITask(AbstractBCITask):
    def __init__(self):
        super().__init__(
            name="Left Hand vs Right Hand MI",
            classes=[BCIClasses.LEFT_HAND_MI, BCIClasses.RIGHT_HAND_MI],
            datasets=[
                BCICompIV2aMDataset,
                BCICompIV2bMDataset,
                Weibo2014MDataset,
                Cho2017MDataset,
                Liu2022MDataset, # gave rather bad results
                Schirrmeister2017MDataset,
                PhysionetMIMDataset,
                Zhou2016MDataset,
                Kaya2018Dataset,
            ],
            subjects_split={
                BCICompIV2aMDataset: {
                    Split.TRAIN: [2, 3, 4, 5, 6, 7, 8, 9],
                    Split.TEST: [1],
                },
                BCICompIV2bMDataset: {
                    Split.TRAIN: [1, 2, 3, 4, 5, 6, 7],
                    Split.TEST: [8, 9],
                },
                Weibo2014MDataset: {
                    Split.TRAIN: [1, 2, 3, 4, 5, 6, 7, 8],
                    Split.TEST: [9, 10],
                },
                Cho2017MDataset: {
                    Split.TRAIN: list(range(1, 10)),
                    Split.TEST: list(range(10, 15)),
                },
                Liu2022MDataset: {
                    Split.TRAIN: list(range(1, 10)),
                    Split.TEST: list(range(10, 15)),
                },
                Schirrmeister2017MDataset: {
                    Split.TRAIN: list(range(1, 12)),
                    Split.TEST: list(range(12, 15)),
                },
                PhysionetMIMDataset: {
                    Split.TRAIN: list(range(1, 88)),
                    Split.TEST: list(range(89, 92)) + list(range(93, 100)) + list(range(101, 110)),
                },
                Zhou2016MDataset: {
                    Split.TRAIN: [],
                    Split.TEST: list(range(1, 5)),
                },
                Kaya2018Dataset: {
                    Split.TRAIN: ["B", "C", "E", "F", "G", "H", "I", "J", "K"],
                    Split.TEST: ["A", "L", "M"],
                }
            },
        )

    def get_metrics(self):
        return lambda y, y_pred: f1_score(y, y_pred.ravel())
