import torch
from torch import nn, Tensor, optim
from torch.nn import functional as F
from torch.autograd import Variable
from typing import Any, Callable, List, Optional, Sequence
from functools import partial
from torchvision import datasets, transforms
from PIL import Image
# 定义了当你使用 from <module> import * 导入某个模块的时候能导出的符号
__all__ = [
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "convnext_large",
]


# 定义随机深度层，简单说就是随机使得一个张量变为全0的张量
class StochasticDepth(nn.Module):

    def __init__(self, p: float, mode: str) -> None:
        super().__init__()
        self.p = p
        self.mode = mode

    # 定义随机深度变换函数的核心函数，简单说就是随机让输入张量变为全0，在外层组合一个类似resnet的短接就实现了随机深度变换函数
    # 当输入张量变为全0时，等效于将本层剔除，实现了n-1层的输出直接送入n+1层，相当于将第n层屏蔽掉了，本函数可实现两种屏蔽模式
    # 第一种是batch模式，就是一个batch内所有样本统一使用同一个随机屏蔽系数，第二种是row模式，就是一个batch内每个样本都有自己的系数
    def stochastic_depth(self, input: Tensor, p: float, mode: str, training: bool = True) -> Tensor:

        if p < 0.0 or p > 1.0:
            raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
        if mode not in ["batch", "row"]:
            raise ValueError(f"mode has to be either 'batch' or 'row', but got {mode}")
        if not training or p == 0.0:
            return input

        survival_rate = 1.0 - p
        if mode == "row":
            size = [input.shape[0]] + [1] * (input.ndim - 1)
        else:
            size = [1] * input.ndim
        noise = torch.empty(size, dtype=input.dtype, device=input.device)  # 基于所选模式，定义随机参数
        noise = noise.bernoulli_(survival_rate)  # 按照概率生成随机参数
        if survival_rate > 0.0:  # 概率为0的不需要做任何操作，但是概率为1的需要除以生存概率，这个类似于dropout
            noise.div_(survival_rate)  # 需要除以生存概率，以保证在平均值统计上，加不加随机深度层时是相同的
        return input * noise

    def forward(self, input: Tensor) -> Tensor:
        return self.stochastic_depth(input, self.p, self.mode, self.training)


# 定义一个卷积+归一化+激活函数层，这个类仅仅在convnext的stem层使用了一次
# torch.nn.Sequential相当于tf2.0中的keras.Sequential()，其实就是以最简单的方式搭建序列模型，不需要写forward()函数，
# 直接以列表形式将每个子模块送进来就可以了，或者也可以使用OrderedDict()或add_module()的形式向模块中添加子模块
class Conv2dNormActivation(torch.nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: Optional[int] = None,
            groups: int = 1,
            norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
            activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
            dilation: int = 1,
            inplace: Optional[bool] = True,
            bias: Optional[bool] = None
    ) -> None:

        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None

        layers = [
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))

        super().__init__(*layers)  # 直接以列表的形式向torch.nn.Sequential中添加子模块


# 定义一个LN类，仅仅在模型的stem层、下采样层和分类层使用了该类，在基本模块中使用的是原生的nn.LayerNorm
class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)  # 将通道维移到最后，对每一个像素的所有维度进行归一化，而非对所有像素的所有维度进行归一化
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)  # 将通道维移到第二位，后面使用时仅仅送入一个参数，所以是对最后一维进行归一化
        return x


# 定义通道转换类，在基本模块中被使用
class Permute(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.Tensor.permute(x, self.dims)


# 定义convnext的最基本模块，包含7*7卷积 + LN + 1*1卷积 + GELU + 1*1卷积 + 层缩放 + 随机深度
class CNBlock(nn.Module):
    def __init__(
            self,
            dim,
            layer_scale: float,
            stochastic_depth_prob: float,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:  # 实际使用时设置为None，所以基本模块使用的都是nn.LayerNorm
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.block = nn.Sequential(  # 用nn.Sequential搭建一个子模块，不需要重写forward()函数
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),  # 深度可分离卷积+反残差结构+替换和减少LN和激活函数
            Permute([0, 2, 3, 1]),  # 实现基本模块的方式有两种，分别是7*7卷积 + Permute + LN + Permute + 1*1卷积 + GELU + 1*1卷积
            norm_layer(dim),  # 或者7*7卷积 + Permute + LN + Linear + GELU + Linear + Permute
            nn.Linear(in_features=dim, out_features=4 * dim, bias=True),  # 经过验证第二种方式运行速度更快，所以本代码使用第二种方式
            nn.GELU(),  # 这里需要强调的是，本代码中的LN并不是标准LN，其并非对一个样本中所有通道的所有像素进行归一化，而是对所有
            nn.Linear(in_features=4 * dim, out_features=dim, bias=True),  # 通道的每个像素分别进行归一化，在torch中并没有直接实现该
            Permute([0, 3, 1, 2]),  # 变换的函数，所以只能通过Permute + LN + Permute组合的方式实现
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")  # 一个batch中不同样本使用不同的随机数

    def forward(self, input: Tensor) -> Tensor:
        result = self.layer_scale * self.block(input)  # 关于layer_scale有个疑问，默认的值为何设置的非常小，而不是1
        result = self.stochastic_depth(result)
        result += input  # 短接，短接与stochastic_depth配合才能实现随机深度的思想
        return result


# 定义整个convnext的每个大模块的配置信息，整个模型的bottleneck层由四个大模块组成，每个大模块又包含很多基本模块
class CNBlockConfig:
    def __init__(self, input_channels: int, out_channels: Optional[int], num_layers: int) -> None:
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.num_layers = num_layers


# 根据配置列表搭建整个convnext模型
class ConvNeXt(nn.Module):
    def __init__(
            self,
            block_setting: List[CNBlockConfig],  # 参数配置列表，实际使用时，包含4个CNBlockConfig
            stochastic_depth_prob: float = 0.0,
            layer_scale: float = 1e-6,
            num_classes: int = 6,
            block: Optional[Callable[..., nn.Module]] = None,  # 实际使用时，采用默认的None
            norm_layer: Optional[Callable[..., nn.Module]] = None,  # 实际使用时，采用默认的None
            **kwargs: Any,  # 实际使用时，没有其它参数输入
    ) -> None:
        super().__init__()

        if not block_setting:
            raise ValueError("The block_setting should not be empty")
        elif not (isinstance(block_setting, Sequence) and all([isinstance(s, CNBlockConfig) for s in block_setting])):
            raise TypeError("The block_setting should be List[CNBlockConfig]")

        if block is None:  # 所以实际使用的就是CNBlock
            block = CNBlock

        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)  # 所以实际使用的就是LayerNorm2d，仅用于模型的stem层、下采样层和分类层

        layers: List[nn.Module] = []

        ### 0. 搭建整个模型的第一层，即Stem层，包含卷积+偏置+LN
        firstconv_output_channels = block_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=4,  # stride和kernel_size均为4
                stride=4,
                padding=0,
                norm_layer=norm_layer,
                activation_layer=None,
                bias=True,
            )
        )
        ### 1. 搭建整个模型的第二部分，即bottleneck层
        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)  # 统计总的基本模块数量，用于计算随机深度的概率值
        stage_block_id = 0  # 这个概率值是越往后面越大，即浅层时尽量不要改变深度
        for cnf in block_setting:  # 遍历四个大模块配置，调整了大模块中小模块的比例
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):  # 遍历每个大模块中的基本模块
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(block(cnf.input_channels, layer_scale, sd_prob))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))  # 用nn.Sequential搭建一个子模块，不需要重写forward()函数
            if cnf.out_channels is not None:  # 定义下采样层，前三个大模块结束后各使用了一次
                layers.append(
                    nn.Sequential(  # 用nn.Sequential搭建一个子模块，不需要重写forward()函数
                        norm_layer(cnf.input_channels),
                        nn.Conv2d(cnf.input_channels, cnf.out_channels, kernel_size=2, stride=2),
                    )
                )
        self.features = nn.Sequential(*layers)  # 用nn.Sequential搭建一个子模块，不需要重写forward()函数
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        ### 2. 搭建最后的分类层
        lastblock = block_setting[-1]
        lastconv_output_channels = (
            lastblock.out_channels if lastblock.out_channels is not None else lastblock.input_channels
        )
        self.classifier = nn.Sequential(  # 用nn.Sequential搭建一个子模块，不需要重写forward()函数
            norm_layer(lastconv_output_channels), nn.Flatten(1), nn.Linear(lastconv_output_channels, num_classes)
        )
        # 初始化参数
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


##############################################################################################################################
## 通过修改配置列表实现不同模型的定义
def convnext_tiny(num_classes: int = 1000, layer_scale: float = 1e-6) -> ConvNeXt:
    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 9),
        CNBlockConfig(768, None, 3),
    ]
    stochastic_depth_prob = 0.1
    return ConvNeXt(block_setting, stochastic_depth_prob, layer_scale, num_classes)


def convnext_small(num_classes: int = 1000, layer_scale: float = 1e-6) -> ConvNeXt:
    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 27),
        CNBlockConfig(768, None, 3),
    ]
    stochastic_depth_prob = 0.4
    return ConvNeXt(block_setting, stochastic_depth_prob, layer_scale, num_classes)


def convnext_base(num_classes: int = 1000, layer_scale: float = 1e-6) -> ConvNeXt:
    block_setting = [
        CNBlockConfig(128, 256, 3),
        CNBlockConfig(256, 512, 3),
        CNBlockConfig(512, 1024, 27),
        CNBlockConfig(1024, None, 3),
    ]
    stochastic_depth_prob = 0.5
    return ConvNeXt(block_setting, stochastic_depth_prob, layer_scale, num_classes)


def convnext_large(num_classes: int = 1000, layer_scale: float = 1e-6) -> ConvNeXt:
    block_setting = [
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 3),
        CNBlockConfig(768, 1536, 27),
        CNBlockConfig(1536, None, 3),
    ]
    stochastic_depth_prob = 0.5
    return ConvNeXt(block_setting, stochastic_depth_prob, layer_scale, num_classes)




# 等比例拉伸图片，多余部分填充value
def resize_padding(image, target_length, value=0):
        h, w = image.size  # 获得原始尺寸
        ih, iw = target_length, target_length  # 获得目标尺寸
        scale = min(iw / w, ih / h)  # 实际拉伸比例
        nw, nh = int(scale * w), int(scale * h)  # 实际拉伸后的尺寸
        image_resized = image.resize((nh, nw), Image.ANTIALIAS)  # 实际拉伸图片
        image_paded = Image.new("RGB", (ih, iw), value)
        dw, dh = (iw - nw) // 2, (ih - nh) // 2
        image_paded.paste(image_resized, (dh, dw, nh + dh, nw + dw))  # 居中填充图片
        return image_paded

# 定义超参数
batch_size = 64
learning_rate = 0.001
epochs = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 变换函数
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# 读取图片并预处理
data = datasets.ImageFolder('data', transform=transforms.Compose([transforms.Lambda(lambda x: resize_padding(x, 224)),
                                                                      transform]))
train_data, test_data = torch.utils.data.random_split(data,
                                                          [int(len(data) * 0.8), len(data) - int(len(data) * 0.8)])
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)


# 建立模型并恢复权重
weight_path = "convnext_tiny-983f1562.pth"
pre_weights = torch.load(weight_path)
model = convnext_tiny()
model.load_state_dict(pre_weights)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 定义训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).to(device), Variable(target).to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

# 定义测试函数
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = Variable(data).to(device), Variable(target).to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# 训练模型
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

# 保存模型
torch.save(model.state_dict(), "convnext_tiny.pth")