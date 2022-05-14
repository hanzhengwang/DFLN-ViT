import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision import models
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.autograd import Variable


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        # init.normal_(m.weight.data, 0, 0.001)
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        init.zeros_(m.bias.data)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class FeatureBlock(nn.Module):
    def __init__(self, input_dim, low_dim, dropout=0.5, relu=True):
        super(FeatureBlock, self).__init__()
        feat_block = []
        feat_block += [nn.Linear(input_dim, low_dim)]
        feat_block += [nn.BatchNorm1d(low_dim)]

        feat_block = nn.Sequential(*feat_block)
        feat_block.apply(weights_init_kaiming)
        self.feat_block = feat_block

    def forward(self, x):
        x = self.feat_block(x)
        return x


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=0.5, relu=True):
        super(ClassBlock, self).__init__()
        classifier = []
        if relu:
            classifier += [nn.LeakyReLU(0.1)]
        if dropout:
            classifier += [nn.Dropout(p=dropout)]

        classifier += [nn.Linear(input_dim, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.classifier = classifier

    def forward(self, x):
        x = self.classifier(x)
        return x

    # Define the ResNet18-based Model


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        # inner_dim = 682
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # b, 65, 1024, heads = 8
        b, n, _, h = *x.shape, self.heads

        # self.to_qkv(x): b, 65, 64*8*3
        # qkv: b, 65, 64*8
        # x = x.unsqueeze(dim=2)       682

        qkv = self.to_qkv(x)
        qkv = qkv.chunk(3, dim=-1)

        # b, 65, 64, 8
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # dots:b, 65, 64, 64
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # attn:b, 65, 64, 64
        attn = dots.softmax(dim=-1)

        # 使用einsum表示矩阵乘法：
        # out:b, 65, 64, 8
        out = torch.einsum('bhij,bhjd->bhid', attn, v)

        # out:b, 64, 65*8
        out = rearrange(out, 'b h n d -> b n (h d)')

        # out:b, 64, 1024
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x
class visible_net_resnet(nn.Module):
    def __init__(self, arch='resnet18'):
        super(visible_net_resnet, self).__init__()
        if arch == 'resnet18':
            model_ft = models.resnet18(pretrained=True)
        elif arch == 'resnet50':
            model_ft = models.resnet50(pretrained=True)

        for mo in model_ft.layer4[0].modules():
            if isinstance(mo, nn.Conv2d):
                mo.stride = (1, 1)

        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.visible = model_ft
        self.dropout = nn.Dropout(p=0.5)
        self.size = [12, 6, 3]
        self.att1 = SAM(256, self.size[0])
        self.att2 = SAM(512, self.size[1])
        self.att3 = SAM(1024, self.size[2])
        self.att4 = SAM(2048, self.size[2])
        self.attc1 = CEM(256, 72, 36)
        self.attc2 = CEM(512, 36, 18)
        self.attc3 = CEM(1024, 18, 9)
        self.attc4 = CEM(2048, 18, 9)
        # self.new4 = New(2048)

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        x = self.visible.layer1(x)
        a1_v = self.att1(x)
        x = self.attc1(x)
        x = self.visible.layer2(x)
        a2_v = self.att2(x)
        x = self.attc2(x)
        x = self.visible.layer3(x)
        a3_v = self.att3(x)
        x = self.attc3(x)
        x = self.visible.layer4(x)
        a4_v = self.att4(x)
        x = self.attc4(x)
        x = x + a1_v + a2_v + a3_v + a4_v

        num_part = 6
        # pool size
        sx = x.size(2) / num_part
        sx = int(sx)
        kx = x.size(2) - sx * (num_part - 1)
        kx = int(kx)
        x = nn.functional.avg_pool2d(x, kernel_size=(kx, x.size(3)), stride=(sx, x.size(3)))
        # x = self.visible.avgpool(x)
        x = x.view(x.size(0), x.size(1), x.size(2))
        # x = self.dropout(x)

        return x


class thermal_net_resnet(nn.Module):
    def __init__(self, arch='resnet18'):
        super(thermal_net_resnet, self).__init__()
        if arch == 'resnet18':
            model_ft = models.resnet18(pretrained=True)
        elif arch == 'resnet50':
            model_ft = models.resnet50(pretrained=True)

        for mo in model_ft.layer4[0].modules():
            if isinstance(mo, nn.Conv2d):
                mo.stride = (1, 1)

        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.thermal = model_ft
        self.dropout = nn.Dropout(p=0.5)
        self.size = [12, 6, 3]
        self.att1 = SAM(256, self.size[0])
        self.att2 = SAM(512, self.size[1])
        self.att3 = SAM(1024, self.size[2])
        self.att4 = SAM(2048, self.size[2])
        self.attc1 = CEM(256, 72, 36)
        self.attc2 = CEM(512, 36, 18)
        self.attc3 = CEM(1024, 18, 9)
        self.attc4 = CEM(2048, 18, 9)
        # self.new4 = New(2048)

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        x = self.thermal.layer1(x)
        a1_t = self.att1(x)
        x = self.attc1(x)
        x = self.thermal.layer2(x)
        a2_t = self.att2(x)
        x = self.attc2(x)
        x = self.thermal.layer3(x)
        a3_t = self.att3(x)
        x = self.attc3(x)
        x = self.thermal.layer4(x)
        a4_t = self.att4(x)
        x = self.attc4(x)
        x = x + a1_t + a2_t + a3_t + a4_t

        num_part = 6  # number of part
        # pool size
        sx = x.size(2) / num_part
        sx = int(sx)
        kx = x.size(2) - sx * (num_part - 1)
        kx = int(kx)
        x = nn.functional.avg_pool2d(x, kernel_size=(kx, x.size(3)), stride=(sx, x.size(3)))
        # x = self.thermal.avgpool(x)
        x = x.view(x.size(0), x.size(1), x.size(2))
        # x = self.dropout(x)

        return x

class embed_net(nn.Module):
    def __init__(self, low_dim, class_num, drop=0.5, arch='resnet50'):
        super(embed_net, self).__init__()
        if arch == 'resnet18':
            self.visible_net = visible_net_resnet(arch=arch)
            self.thermal_net = thermal_net_resnet(arch=arch)
            pool_dim = 512
        elif arch == 'resnet50':
            self.visible_net = visible_net_resnet(arch=arch)
            self.thermal_net = thermal_net_resnet(arch=arch)
            pool_dim = 2048

        self.feature1 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature2 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature3 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature4 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature5 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature6 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.classifier1 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier2 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier3 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier4 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier5 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier6 = ClassBlock(low_dim, class_num, dropout=drop)
        self.dist = nn.MSELoss(reduction='sum')
        self.l2norm = Normalize(2)

    def forward(self, x1, x2, modal=0):
        if modal == 0:
            x1 = self.visible_net(x1)
            x2 = self.thermal_net(x2)
            x1 = x1.chunk(6, 2)
            x1_0 = x1[0].contiguous().view(x1[0].size(0), -1)
            x1_1 = x1[1].contiguous().view(x1[1].size(0), -1)
            x1_2 = x1[2].contiguous().view(x1[2].size(0), -1)
            x1_3 = x1[3].contiguous().view(x1[3].size(0), -1)
            x1_4 = x1[4].contiguous().view(x1[4].size(0), -1)
            x1_5 = x1[5].contiguous().view(x1[5].size(0), -1)

            x2 = x2.chunk(6, 2)
            x2_0 = x2[0].contiguous().view(x2[0].size(0), -1)
            x2_1 = x2[1].contiguous().view(x2[1].size(0), -1)
            x2_2 = x2[2].contiguous().view(x2[2].size(0), -1)
            x2_3 = x2[3].contiguous().view(x2[3].size(0), -1)
            x2_4 = x2[4].contiguous().view(x2[4].size(0), -1)
            x2_5 = x2[5].contiguous().view(x2[5].size(0), -1)
            x_0 = torch.cat((x1_0, x2_0), 0)
            x_1 = torch.cat((x1_1, x2_1), 0)
            x_2 = torch.cat((x1_2, x2_2), 0)
            x_3 = torch.cat((x1_3, x2_3), 0)
            x_4 = torch.cat((x1_4, x2_4), 0)
            x_5 = torch.cat((x1_5, x2_5), 0)
            # loss = self.dist(v_mask, t_mask)
        elif modal == 1:
            x = self.visible_net(x1)
            x = x.chunk(6, 2)
            x_0 = x[0].contiguous().view(x[0].size(0), -1)
            x_1 = x[1].contiguous().view(x[1].size(0), -1)
            x_2 = x[2].contiguous().view(x[2].size(0), -1)
            x_3 = x[3].contiguous().view(x[3].size(0), -1)
            x_4 = x[4].contiguous().view(x[4].size(0), -1)
            x_5 = x[5].contiguous().view(x[5].size(0), -1)
        elif modal == 2:
            x = self.thermal_net(x2)
            x = x.chunk(6, 2)
            x_0 = x[0].contiguous().view(x[0].size(0), -1)
            x_1 = x[1].contiguous().view(x[1].size(0), -1)
            x_2 = x[2].contiguous().view(x[2].size(0), -1)
            x_3 = x[3].contiguous().view(x[3].size(0), -1)
            x_4 = x[4].contiguous().view(x[4].size(0), -1)
            x_5 = x[5].contiguous().view(x[5].size(0), -1)

        y_0 = self.feature1(x_0)
        y_1 = self.feature2(x_1)
        y_2 = self.feature3(x_2)
        y_3 = self.feature4(x_3)
        y_4 = self.feature5(x_4)
        y_5 = self.feature6(x_5)
        # y = self.feature(x)
        out_0 = self.classifier1(y_0)
        out_1 = self.classifier2(y_1)
        out_2 = self.classifier3(y_2)
        out_3 = self.classifier4(y_3)
        out_4 = self.classifier5(y_4)
        out_5 = self.classifier6(y_5)
        # out = self.classifier(y)
        if self.training:
            return (out_0, out_1, out_2, out_3, out_4, out_5), (
                self.l2norm(y_0), self.l2norm(y_1), self.l2norm(y_2), self.l2norm(y_3), self.l2norm(y_4),
                self.l2norm(y_5))
        else:
            x_0 = self.l2norm(x_0)
            x_1 = self.l2norm(x_1)
            x_2 = self.l2norm(x_2)
            x_3 = self.l2norm(x_3)
            x_4 = self.l2norm(x_4)
            x_5 = self.l2norm(x_5)
            x = torch.cat((x_0, x_1, x_2, x_3, x_4, x_5), 1)
            y_0 = self.l2norm(y_0)
            y_1 = self.l2norm(y_1)
            y_2 = self.l2norm(y_2)
            y_3 = self.l2norm(y_3)
            y_4 = self.l2norm(y_4)
            y_5 = self.l2norm(y_5)
            y = torch.cat((y_0, y_1, y_2, y_3, y_4, y_5), 1)
            return x, y

class SAM(nn.Module):
    def __init__(self, channels, size):
        super().__init__()
        self.transformer = Transformer(dim=2048, depth=1, heads=1, dim_head=32, mlp_dim=16, dropout=0.1)
        self.attention_weight = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.1)

        self.pos_embedding = nn.Parameter(torch.randn(1, 18, 2048))
        self.patch_embedding = nn.Conv2d(channels, 2048, kernel_size=size, stride=size)

    def forward(self, x):
        last_block = x
        b, c, h, w = last_block.size()
        input = self.patch_embedding(x)  # 32, 2048, 6, 3
        input = input.view(b, 2048, -1).transpose(1, 2)  # 32, 18, 2048
        input = input + self.pos_embedding[:, :(2048)]
        output = self.transformer(input)  # 32, 18, 2048
        output = output.view(b, 2048, 6, 3)
        output = F.interpolate(output, (18, 9), mode='bilinear', align_corners=True)

        return output

class CEM(nn.Module):
    def __init__(self, channels, H, W):
        super().__init__()
        self.transformer = Transformer(dim=1, depth=1, heads=1, dim_head=32, mlp_dim=16, dropout=0.1)
        self.attention_weight = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.1)
        self.pos_embedding = nn.Parameter(torch.randn(1, channels, 1))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        b, c, h, w = x.size()  # 32, 256, 72, 36
        input = self.gap(x).squeeze(-1)  # 32， 256， 72*36=2592

        _, c, _ = input.shape

        input = input + self.pos_embedding[:, :(c)]
        input = self.dropout(input)

        output = self.transformer(input)  # 32, 256, 1
        output = torch.unsqueeze(output, dim=3)  # 32, 256, 1, 1

        weight = torch.sigmoid(output)  # 32, 256, 1, 1

        # w = weight.repeat(1, 1, h, w)

        return weight * x