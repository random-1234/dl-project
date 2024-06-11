import os
import io
import torch
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
from torchvision import transforms
from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt
import time
from torch import nn, optim

app = Flask(__name__)

transform = transforms.Compose([
    transforms.Resize((256, 256), Image.BICUBIC),
    transforms.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PatchDiscriminator(nn.Module):
    def __init__(self, input_c, num_filters=64, n_down=3):
        super().__init__()
        model = [self.get_layers(input_c, num_filters, norm=False)]
        model += [self.get_layers(num_filters * 2 ** i, num_filters * 2 ** (i + 1), s=1 if i == (n_down-1) else 2)
                  for i in range(n_down)]
        model += [self.get_layers(num_filters * 2 ** n_down, 1, s=1, norm=False, act=False)]
        self.model = nn.Sequential(*model)

    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True, act=True):
        layers = [nn.Conv2d(ni, nf, k, s, p, bias=not norm)]
        if norm: layers += [nn.BatchNorm2d(nf)]
        if act: layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def load_models(generator_path, discriminator_path):
    resnet18_model = resnet18(pretrained=True)
    body = create_body(resnet18_model, n_in=1, cut=-2)
    net_G = DynamicUnet(body, 2, (256, 256)).to(device)
    net_G.load_state_dict(torch.load(generator_path, map_location=device))

    net_D = PatchDiscriminator(3).to(device)
    net_D.load_state_dict(torch.load(discriminator_path, map_location=device))
    
    return net_G, net_D

net_G, net_D = load_models('models/generator.pth', 'models/discriminator.pth')

class MainModel:
    def __init__(self, net_G, net_D):
        self.net_G = net_G
        self.net_D = net_D

    def setup_input(self, data):
        self.L = data['L'].to(device)
        self.ab = data['ab'].to(device)

    def forward(self):
        self.fake_color = self.net_G(self.L)

model = MainModel(net_G=net_G, net_D=net_D)

def lab_to_rgb(L, ab):
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)

def save_image(image, path):
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        img = Image.open(file.stream).convert("RGB")
        img = transform(img)

        img_lab = rgb2lab(img.permute(1, 2, 0).cpu().numpy()).astype("float32")
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1.
        ab = img_lab[[1, 2], ...] / 110.

        L = L.unsqueeze(0)
        ab = ab.unsqueeze(0)

        data = {'L': L, 'ab': ab}
        model.setup_input(data)
        model.forward()

        fake_color = model.fake_color.detach()
        L = model.L
        fake_imgs = lab_to_rgb(L, fake_color)[0]
        real_imgs = lab_to_rgb(L, ab)[0]
        gray_imgs = (L[0, 0].cpu().numpy() + 1) * 50
        gray_imgs = gray_imgs.astype(np.uint8)

        save_image(fake_imgs, 'static/fake_color.png')
        save_image(real_imgs, 'static/real_color.png')
        Image.fromarray(gray_imgs).convert('L').save('static/grayscale.png')

        return redirect(url_for('results'))

@app.route('/results')
def results():
    return render_template('results.html')

if __name__ == '__main__':
    app.run(debug=True)
