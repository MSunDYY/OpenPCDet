from .partA2_head import PartA2FCHead
from .pointrcnn_head import PointRCNNHead
from .pvrcnn_head import PVRCNNHead
from .second_head import SECONDHead
from .voxelrcnn_head import VoxelRCNNHead
from .roi_head_template import RoIHeadTemplate
from .mppnet_head import MPPNetHead
from .mppnet_memory_bank_e2e import MPPNetHeadE2E
from .velocity_head import VelocityHead
from .denet_head import DENetHead
from .msf_head import MSFHead
from .denet_head2 import DENet2Head
from .denet_head3 import DENet3Head
from .denet_head4 import DENet4Head
from .denet_head5 import DENet5Head
__all__ = {
    'RoIHeadTemplate': RoIHeadTemplate,
    'PartA2FCHead': PartA2FCHead,
    'PVRCNNHead': PVRCNNHead,
    'SECONDHead': SECONDHead,
    'PointRCNNHead': PointRCNNHead,
    'VoxelRCNNHead': VoxelRCNNHead,
    'MPPNetHead': MPPNetHead,
    'MPPNetHeadE2E': MPPNetHeadE2E,
    'VelocityHead': VelocityHead,
    'DENetHead': DENetHead,
    'MSFHead': MSFHead,
    'DENet2Head': DENet2Head,
    'DENet3Head':DENet3Head,
    'DENet4Head':DENet4Head,
    'DENet5Head':DENet5Head,
}
