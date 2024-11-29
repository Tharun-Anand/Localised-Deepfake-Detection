import glob
import os
import sys

from numpy import ndarray
from typing import List, Any, TypeVar, Union, Dict, Type, Optional, Tuple, Sequence,Callable
from torch import Tensor

import numpy as np
import yaml
from einops import rearrange

import cv2
import torch
import torchvision
import torchvision.transforms as VT
from torch.nn import functional as F
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings("ignore")



import torch.nn as nn
import math
from einops.layers.torch import Rearrange

Shape = Union[torch.Size, List[int], Tuple[int, ...]]
ModuleFactory = Union[Callable[[], nn.Module], Callable[[int], nn.Module]]