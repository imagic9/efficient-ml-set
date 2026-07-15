"""Model definitions and the class-map contract.

MobileNetV2 width 1.0 with a 16-output head: 14 animals + `car` + `empty`. The
class order is frozen in `configs/data/classes.yaml` and must never be inferred
from dictionary iteration order — the CCT category IDs are non-contiguous.
"""
