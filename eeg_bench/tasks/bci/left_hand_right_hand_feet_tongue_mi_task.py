from .abstract_bci_task import AbstractBCITask
from ...datasets.bci import (
    BCICompIV2aMDataset,
    Kaya2018Dataset,
)
from ...enums.bci_classes import BCIClasses
from sklearn.metrics import f1_score
from ...enums.split import Split
import os
import json


class LeftHandvRightHandvFeetvTongueMITask(AbstractBCITask):
    def __init__(self):
        super().__init__(
            name="Left Hand vs Right Hand vs Feet vs Tongue MI",
            classes=[
                BCIClasses.LEFT_HAND_MI,
                BCIClasses.RIGHT_HAND_MI,
                BCIClasses.FEET_MI,
                BCIClasses.TONGUE_MI,
            ],
            datasets=[
                BCICompIV2aMDataset,
                Kaya2018Dataset,
            ],
            subjects_split={
                BCICompIV2aMDataset: {
                    Split.TRAIN: [1, 2, 3, 4, 5, 6, 7],
                    Split.TEST: [8, 9],
                },
                Kaya2018Dataset: {
                    Split.TRAIN: ["B", "C", "E", "F", "G", "H", "I", "J", "K"], 
                    Split.TEST: ["A", "L", "M" ], 
                },
            },
        )

    def get_metrics(self):
        return lambda y, y_pred: f1_score(y, y_pred.ravel())
