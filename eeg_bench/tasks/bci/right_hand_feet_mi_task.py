from .abstract_bci_task import AbstractBCITask
from ...datasets.bci import (
    Weibo2014MDataset,
    Schirrmeister2017MDataset,
    PhysionetMIMDataset,
    BCICompIV2aMDataset,
    Barachant2012MDataset,
    Zhou2016MDataset,
    Faller2012MDataset,
    Scherer2015MDataset,
    Kaya2018Dataset,
)
from ...enums.bci_classes import BCIClasses
from sklearn.metrics import f1_score
from ...enums.split import Split
import os
import json


class RightHandvFeetMITask(AbstractBCITask):
    def __init__(self):
        super().__init__(
            name="Right Hand vs Feet MI",
            classes=[BCIClasses.RIGHT_HAND_MI, BCIClasses.FEET_MI],
            datasets=[
                Weibo2014MDataset,
                Schirrmeister2017MDataset,
                PhysionetMIMDataset,
                BCICompIV2aMDataset,
                Barachant2012MDataset,
                Zhou2016MDataset,
                Faller2012MDataset,
                Scherer2015MDataset,
                Kaya2018Dataset,
            ],
            subjects_split={
                Weibo2014MDataset: {
                    Split.TRAIN: [1, 2, 3, 4, 5, 6, 7, 8],
                    Split.TEST: [9, 10],
                },
                Schirrmeister2017MDataset: {
                    Split.TRAIN: list(range(1, 12)),
                    Split.TEST: list(range(12, 15)),
                },
                PhysionetMIMDataset: {
                    Split.TRAIN: list(range(1, 88)),
                    Split.TEST: list(range(89, 92)) + list(range(93, 100)) + list(range(101, 110)),
                },
                BCICompIV2aMDataset: {
                    Split.TRAIN: [1, 2, 3, 4, 5, 6, 7],
                    Split.TEST: [8, 9],
                },
                Barachant2012MDataset: {
                    Split.TRAIN: [1, 2, 3, 4, 5, 6],
                    Split.TEST: [7, 8],
                },
                Zhou2016MDataset: {
                    Split.TRAIN: [],
                    Split.TEST: list(range(1, 5)),
                },
                Faller2012MDataset: {
                    Split.TRAIN: list(range(1, 10)),
                    Split.TEST: list(range(10, 12)),
                },
                Scherer2015MDataset: {
                    Split.TRAIN: list(range(1, 7)),
                    Split.TEST: list(range(7, 9)),
                },
                Kaya2018Dataset: {
                    Split.TRAIN: ["B", "C", "E", "F", "G", "H", "I", "J", "K"],
                    Split.TEST: ["A", "L", "M"],
                },
            },
        )

    def get_metrics(self):
        return lambda y, y_pred: f1_score(y, y_pred.ravel())
