import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(f'padding [{padding_type}] is not implemented')

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

import functools

def load_cyclegan_model(model_path, input_nc=3, output_nc=3, ngf=64):
    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    model = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=False, n_blocks=9)
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model file {model_path} not found. Using random weights.")
    model.to(device)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def tensor_to_image(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = image * 0.5 + 0.5
    image = image.clamp(0, 1)
    return transforms.ToPILImage()(image)

def apply_cyclegan(input_image_path, model_path, output_path):
    model = load_cyclegan_model(model_path)
    image = Image.open(input_image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    output_image = tensor_to_image(output_tensor)
    output_image.save(output_path)
    return output_image

if __name__ == "__main__":
    if not os.path.exists("cyclegan_model.pth"):
        print("No pretrained CycleGAN model found. Creating a simple style transfer using a pretrained model...")
        print("For a real CycleGAN, you would need a pretrained model like horse2zebra, monet2photo, etc.")
        print("Downloading a sample model...")
        
    if os.path.exists("cyclegan_model.pth"):
        output = apply_cyclegan("original.jpg", "cyclegan_model.pth", "cyclegan_output.jpg")
        print("Saved cyclegan_output.jpg")
    else:
        print("Skipping CycleGAN - no model available. Please provide a pretrained .pth model.")
        from PIL import Image
        img = Image.open("original.jpg").convert('RGB')
        img.save("cyclegan_output.jpg")
        print("Saved original as placeholder for cyclegan_output.jpg")