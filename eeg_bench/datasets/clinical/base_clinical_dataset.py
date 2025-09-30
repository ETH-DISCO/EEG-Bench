from ..abstract_dataset import AbstractDataset
from typing import Tuple, Optional, Sequence
from ...enums.clinical_classes import ClinicalClasses


class BaseClinicalDataset(AbstractDataset):
    def __init__(
        self,
        name: str,
        target_classes: Sequence[ClinicalClasses],
        available_classes: Sequence[ClinicalClasses],
        subjects: Sequence[int],
        sampling_frequency: int,
        channel_names: Sequence[str],
        target_channels: Optional[Sequence[str]] = None,
        target_frequency: Optional[int] = None,
        preload: bool = False,
    ):
        super().__init__(
            target_classes=target_classes, # type: ignore
            subjects=subjects,
        )
        self.name = name
        if target_classes is not None:
            assert set(target_classes) <= set(available_classes), "Target classes must be a subset of the available classes"
        self.available_classes = available_classes
        self._channel_names = channel_names
        # Default value, to be overridden by subclasses
        self._sampling_frequency = sampling_frequency
        # Default value, to be overridden by subclasses
        if target_channels is not None:
            assert all(
                [channel in self._channel_names for channel in target_channels]
            ), "Target channels must be a subset of the available channels"
        self._target_channels = target_channels
        self._target_frequency = target_frequency
        self._preload = preload
        # TODO make this more generic with a config file and parameters

    def __getitem__(self, index):
        if self.data is None:
            self.load_data()
        return self.data[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.data)

    def __str__(self) -> str:
        return self.name

    def _download(self):
        pass
