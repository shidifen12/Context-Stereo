from .kitti_dataset import KITTIDataset
from .sceneflow_dataset import SceneFlowDatset
from .sota_dataset import SOTADataset

__datasets__ = {
    "sceneflow": SceneFlowDatset,
    "kitti": KITTIDataset,
    "sota": SOTADataset
}
