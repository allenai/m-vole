"""
=================
This is the basic model containing place holder/implementation for necessary functions.

All the models in this project inherit from this class
=================
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18 as torchvision_resnet18


class FeatureLearnerModule(nn.Module):

    def __init__(self):
        super().__init__()
        resnet_model = torchvision_resnet18(pretrained=True)
        del resnet_model.fc
        self.resnet = resnet_model
        self.intermediate_features = None



    def resnet_features(self, x):

        result = []

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        result.append(x)
        x = self.resnet.maxpool(x)




        x = self.resnet.layer1(x)
        result.append(x)

        x = self.resnet.layer2(x)
        result.append(x)


        x = self.resnet.layer3(x)
        result.append(x)


        x = self.resnet.layer4[0](x)
        x = self.resnet.layer4[1](x)
        result.append(x)
        x = self.resnet.avgpool(x)


        x = x.view(x.shape[0], 512)

        self.intermediate_features = result

        return x

    def get_feature(self, images):

        shape = list(images.shape)
        if len(shape) == 4:
            features = self.resnet_features(images)
        else:
            batchsize = shape[0]
            sequence_length = shape[1]
            images = images.contiguous().view([batchsize * sequence_length] + shape[2:])
            features = self.resnet_features(images)
            features = features.view(batchsize, sequence_length, 512)
        return features


    def forward(self, images):
        self.intermediate_features = None #just a sanity check that they are reinitialized each time
        return self.get_feature(images)
