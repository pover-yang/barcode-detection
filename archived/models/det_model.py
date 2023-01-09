import torch.nn as nn
from models.fastrcnn import FasterRCNN
from torchvision.models.detection import fasterrcnn_resnet50_fpn


class BarcodeDet(nn.Module):
    def __init__(self, backbone, num_classes, transform):
        super(BarcodeDet, self).__init__()
        self.model = FasterRCNN(backbone, num_classes, transform)

    def forward(self, images, targets):
        losses, detections = self.model(images, targets)
        return losses, detections