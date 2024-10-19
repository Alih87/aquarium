from mmcv.ops import DeformConv2d
import torchvision, torch
from torchvision.transforms import transforms
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.transform import GeneralizedRCNNTransform
import torch.nn as nn
import torch.functional as F
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import FasterRCNN, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torch.jit.annotations import List, Tuple, Dict, Optional


# class DownSample(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DownSample, self).__init__()
#         self.offsets = nn.Conv2d(in_channels, 2, kernel_size=(1, 1), stride=(2, 2), bias=True)
#         self.conv = DeformConv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(2, 2), dilation=(1, 1), groups=1, deformable_groups=1)
#         self.bn = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

#     def forward(self, x):
#         offsets = self.offsets(x)
#         out = self.conv(x, offsets)
#         out = self.bn(out)
#         return out

# class CustomResNetBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, downsample=False):
#         super(CustomResNetBlock, self).__init__()
#         self.conv1_offsets = nn.Conv2d(in_channels, 18, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=True)
#         self.conv1 = DeformConv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1), groups=1, deformable_groups=1)
#         self.bn1 = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2_offsets = nn.Conv2d(out_channels, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
#         self.conv2 = DeformConv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1, deformable_groups=1)
#         self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         self.conv3_offsets = nn.Conv2d(out_channels, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
#         self.conv3 = DeformConv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1, deformable_groups=1)
#         self.bn3 = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         self.downsample = downsample
#         if self.downsample:
#             self.down_sample = DownSample(in_channels, out_channels)

#     def forward(self, x):
#         identity = x
#         offsets = self.conv1_offsets(x)
#         out = self.conv1(x, offsets)
#         out = self.bn1(out)
#         out = self.relu(out)
#         offsets = self.conv2_offsets(out)
#         out = self.conv2(out, offsets)
#         out = self.bn2(out)
#         if self.downsample:
#             identity = self.down_sample(identity)
#             out += identity
#         out = self.relu(out)
#         return out

# Custom ResNet Block
class CustomResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(CustomResNetBlock, self).__init__()
        # Main path: Deformable Conv + Offsets
        self.conv1_offsets = nn.Conv2d(in_channels, 18, kernel_size=(3, 3), stride=(2 if downsample else 1), padding=1, bias=True)
        self.conv1 = DeformConv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(2 if downsample else 1), padding=1, dilation=1, groups=1, deformable_groups=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2_offsets = nn.Conv2d(out_channels, 18, kernel_size=(3, 3), stride=1, padding=1, bias=True)
        self.conv2 = DeformConv2d(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1, deformable_groups=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Identity path (skip connection): Ensure matching dimensions
        self.downsample = downsample or (in_channels != out_channels)
        if self.downsample:
            # Apply downsampling to match both spatial size and channels
            self.down_sample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(2 if downsample else 1), bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down_sample = None

    def forward(self, x):
        identity = x

        # Main path
        offsets = self.conv1_offsets(x)
        out = self.conv1(x, offsets)
        out = self.bn1(out)
        out = self.relu(out)

        offsets = self.conv2_offsets(out)
        out = self.conv2(out, offsets)
        out = self.bn2(out)

        # If downsampling is required, apply it to the identity tensor
        if self.down_sample is not None:
            identity = self.down_sample(identity)

        # Add the identity (skip connection) to the output
        out += identity
        out = self.relu(out)

        return out


# Custom ResNet Backbone
class ResNet50_CNN(nn.Module):
    def __init__(self):
        super(ResNet50_CNN, self).__init__()
        self.resnet = torchvision.models.resnet50(weights=None)
        resnet_pretrained = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
        pretrained_dict = resnet_pretrained.state_dict()
        model_dict = self.resnet.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'fc' not in k}
        model_dict.update(pretrained_dict)
        self.resnet.load_state_dict(model_dict)

        for param in self.resnet.parameters():
            param.requires_grad = False

        # Replacing layer3 and layer4 with custom blocks
        self.resnet.layer3 = nn.Sequential(
            CustomResNetBlock(512, 1024, downsample=True),
            CustomResNetBlock(1024, 1024)
        )
        self.resnet.layer4 = nn.Sequential(
            CustomResNetBlock(1024, 2048, downsample=True),
            CustomResNetBlock(2048, 2048)
        )

        # Initialize new layers
        self.resnet.layer3.apply(weights_init)
        self.resnet.layer4.apply(weights_init)

        # Keep up to layer4, we stop before the adaptive pool and fc
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-2])

        # Explicitly set the number of output channels (for Faster R-CNN to use)
        self.out_channels = 2048  # Set to 2048 for ResNet-50

    def forward(self, x):
        x = self.feature_extractor(x)
        return {"0": x}  # Return a dictionary of feature maps for the RoI pooling

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    # elif isinstance(m, DeformConv2d):
    #     nn.init.kaiming_normal_(m.weight)
    #     nn.init.constant_(m.bias, 0)

# Faster R-CNN model with the custom backbone
def ResNet50_FasterRCNN(NUM_CLASSES):
    # Load custom backbone
    resnet = ResNet50_CNN()
    
    # Anchor sizes based on object size in the dataset
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
    
    # RoI pooling - adjust the output size according to backbone feature map
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)
    
    # Faster R-CNN model
    faster_rcnn = FasterRCNN(
        resnet, 
        num_classes=NUM_CLASSES, 
        rpn_anchor_generator=anchor_generator, 
        box_roi_pool=roi_pooler
    )
    
    # Replace the box predictor (head) for the number of classes
    in_features = faster_rcnn.roi_heads.box_predictor.cls_score.in_features
    faster_rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    
    return faster_rcnn

# Testing the model
if __name__ == '__main__':
    NUM_CLASSES = 8
    model = ResNet50_FasterRCNN(NUM_CLASSES)
    
    model.eval()
    model.to("cuda")
    
    # Input: Normalized image similar to ImageNet normalization
    img = torch.rand(2, 3, 1024, 768).to("cuda")
    
    # Forward pass
    with torch.no_grad():
        out = model([img[0]])  # Faster R-CNN expects a list of images
    print(out)