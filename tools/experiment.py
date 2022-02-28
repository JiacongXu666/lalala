import torch
import _init_paths
import models
import datasets
from configs import config
from configs import update_config
from utils.criterion import CrossEntropy, OhemCrossEntropy, JointBondaryLoss
from utils.function import train, validate
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel

filename = 'C:/Files/2022_github/pid_test/output/cityscapes/diff_ddrnet23_single/checkpoint.pth.tar'
pretrained_state = torch.load(filename, map_location='cpu') 
model = models.diff_ddrnet_23.DualResNet(models.diff_ddrnet_23.BasicBlock, [2, 2, 2, 2], num_classes=19, planes=64, spp_planes=128, head_planes=128, augment=True)
model = torch.nn.DataParallel(model)
model.module.load_state_dict({k.replace('model.', ''): v for k, v in pretrained_state['state_dict'].items() if k.startswith('model.')})