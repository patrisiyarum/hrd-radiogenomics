"""hrd-radiogenomics — CT → HRD 3D CNN transfer learning on TCGA-OV × TCIA."""

__version__ = "0.0.1"

# The preprocessing contract is intentionally identical to the one in
# drug-cell-viz's apps/api/src/api/services/radiogenomics.py so that a
# checkpoint trained here can be dropped into drug-cell-viz without any
# re-preprocessing logic.
TARGET_SHAPE: tuple[int, int, int] = (96, 96, 96)
HU_WINDOW: tuple[float, float] = (-200.0, 250.0)
