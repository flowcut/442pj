import torch
import torch.nn as nn
import numpy as np
from IPython import embed
import csv

from .base_color import *


class ECCVGenerator(BaseColor):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(ECCVGenerator, self).__init__()

        model1 = [
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True), ]
        model1 += [nn.ReLU(True), ]
        model1 += [
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True), ]
        model1 += [nn.ReLU(True), ]
        model1 += [norm_layer(64), ]

        model2 = [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1,
                            bias=True), ]
        model2 += [nn.ReLU(True), ]
        model2 += [nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1,
                             bias=True), ]
        model2 += [nn.ReLU(True), ]
        model2 += [norm_layer(128), ]

        model3 = [nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1,
                            bias=True), ]
        model3 += [nn.ReLU(True), ]
        model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,
                             bias=True), ]
        model3 += [nn.ReLU(True), ]
        model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1,
                             bias=True), ]
        model3 += [nn.ReLU(True), ]
        model3 += [norm_layer(256), ]

        model4 = [nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1,
                            bias=True), ]
        model4 += [nn.ReLU(True), ]
        model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1,
                             bias=True), ]
        model4 += [nn.ReLU(True), ]
        model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1,
                             bias=True), ]
        model4 += [nn.ReLU(True), ]
        model4 += [norm_layer(512), ]

        model5 = [
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2,
                      bias=True), ]
        model5 += [nn.ReLU(True), ]
        model5 += [
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2,
                      bias=True), ]
        model5 += [nn.ReLU(True), ]
        model5 += [
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2,
                      bias=True), ]
        model5 += [nn.ReLU(True), ]
        model5 += [norm_layer(512), ]

        model6 = [
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2,
                      bias=True), ]
        model6 += [nn.ReLU(True), ]
        model6 += [
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2,
                      bias=True), ]
        model6 += [nn.ReLU(True), ]
        model6 += [
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2,
                      bias=True), ]
        model6 += [nn.ReLU(True), ]
        model6 += [norm_layer(512), ]

        model7 = [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1,
                            bias=True), ]
        model7 += [nn.ReLU(True), ]
        model7 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1,
                             bias=True), ]
        model7 += [nn.ReLU(True), ]
        model7 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1,
                             bias=True), ]
        model7 += [nn.ReLU(True), ]
        model7 += [norm_layer(512), ]

        model8 = [
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1,
                               bias=True), ]
        model8 += [nn.ReLU(True), ]
        model8 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,
                             bias=True), ]
        model8 += [nn.ReLU(True), ]
        model8 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,
                             bias=True), ]
        model8 += [nn.ReLU(True), ]

        model8 += [nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0,
                             bias=True), ]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)

        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0,
                                   dilation=1, stride=1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, input_l):
        conv1_2 = self.model1(self.normalize_l(input_l))
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)
        out = self.softmax(conv8_3)
        print('out shape:', out.shape)
        return out
        # raw, softmax_raw

        # out_reg = self.model_out(out)

        # return self.unnormalize_ab(self.upsample4(out_reg))


class MultinomialScore(nn.Module):
    def __init__(self, weight_term_file, h=256, w=256, q=313):
        super(MultinomialScore, self).__init__()
        self.weight_term = np.zeros(313)
        self.H = h
        self.W = w
        self.Q = q
        with open(weight_term_file) as current_file:
            reader = csv.DictReader(current_file)
            for index, row in enumerate(reader):
                self.weight_term[index] = float(row[3])
        self.weight_term = torch.tensor(self.weight_term)

    def forward(self, origin, origin_soft, pred):
        """
        :param origin:
            (B, Channel=1, H, W)
        :param origin_soft:
            (B, Channel=313, H, W)
        :param pred:
            (B, Channel=313, H, W)
        :return:
            loss
        """
        weighted_origin = np.zeros(origin.shape)
        for _b in origin.shape[0]:
            for _h in origin.shape[2]:
                for _w in origin.shape[3]:
                    weighted_origin[_b, 1, _h, _w] = self.weight_term[
                        [_b, 1, _h, _w]]
        return (weighted_origin * (origin_soft * torch.log(pred)).sum(
            axis=1)).sum()

def eccv16(pretrained=True):
    model = ECCVGenerator()
    if (pretrained):
        import torch.utils.model_zoo as model_zoo
        model.load_state_dict(model_zoo.load_url(
            'https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth',
            map_location='cpu', check_hash=True))
    return model
