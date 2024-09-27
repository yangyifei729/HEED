from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, NewType, Tuple
import torch
from torch.nn.utils.rnn import pad_sequence

class DataCollator(ABC):
    """
    A `DataCollator` is responsible for batching
    and pre-processing samples of data as requested by the training loop.
    """

    @abstractmethod
    def collate_batch(self) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        pass
InputDataClass = NewType("InputDataClass", Any)
