from typing import Callable, Dict, Mapping, Sequence, Union

from .hooks import FeatureExtractor
from .image_utils import cv2_imshow, get_image_from_url, read_image_to_tensor
from .update_module_state import load_from_ultralytics


__all__ = [
    "FeatureExtractor",
    "cv2_imshow",
    "get_image_from_url",
    "get_callable_dict",
    "load_from_ultralytics",
    "read_image_to_tensor",
]


def get_callable_name(fn_or_class: Union[Callable, object]) -> str:
    return getattr(fn_or_class, "__name__", fn_or_class.__class__.__name__).lower()


def get_callable_dict(fn: Union[Callable, Mapping, Sequence]) -> Union[Dict, Mapping]:
    if isinstance(fn, Mapping):
        return fn
    elif isinstance(fn, Sequence):
        return {get_callable_name(f): f for f in fn}
    elif callable(fn):
        return {get_callable_name(fn): fn}
