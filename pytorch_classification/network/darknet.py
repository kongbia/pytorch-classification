import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ['Darknet']

def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['): # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs

def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    # hyperparams = module_defs.pop(0)
    # output_filters = [int(hyperparams["channels"])]
    output_filters = [3]
    module_list = nn.ModuleList()
    for module_i, module_def in enumerate(module_defs): 
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)

        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{module_i}", EmptyLayer())

        elif module_def["type"] == "shortcut":
            # filters = output_filters[1:][int(module_def["from"])]
            filters = output_filters[1:][-1]
            modules.add_module(f"shortcut_{module_i}", EmptyLayer())
        
        elif module_def["type"] == "dropout":
            probability = float(module_def["probability"])
            modules.add_module(f"dropout_{module_i}", nn.Dropout(probability))
    
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return module_list, output_filters[-1]


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


def darknet_shortcut(input, residual):
    '''
    same as darknet c version
    input->x, residual->f(x)
    '''
    ic, ih, iw = input.shape[1:]
    oc, oh, ow = residual.shape[1:]
    stride, sample = ih//oh, oh//ih
    assert (stride == iw//ow and sample == ow//iw)
    stride = 1 if stride < 1 else stride
    sample = 1 if sample < 1 else sample
    minc, minh, minw = min(ic, oc), min(ih, oh), min(iw, ow)
    output = residual.clone().requires_grad_()  
    output[:, :minc, :minh, :minw] = residual[:, :minc, :oh:sample, :ow:sample] +\
                                        input[:, :minc, :ih:stride, :iw:stride]
    
    return output


class Darknet(nn.Module):
    def __init__(self, config_path, num_classes=1000):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.module_list, last_filters = create_modules(self.module_defs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(last_filters, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        layer_outputs = []
        for module_def, module in zip(self.module_defs, self.module_list):
            if module_def["type"] in ["convolutional", "upsample", "maxpool", "dropout"]:
                x = module(x)
            elif module_def["type"] == "route":
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = darknet_shortcut(layer_outputs[layer_i], layer_outputs[-1])
                # x = layer_outputs[-1] + layer_outputs[layer_i]
            layer_outputs.append(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


if __name__ == '__main__':
    net = Darknet(config_path='configs/darknet_cfg/darknet_p.cfg', num_classes=10).cuda()
    input = torch.FloatTensor(2,3,32,32).cuda()
    output = net(input)
    print(net)
    # print(output.shape)
    print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))