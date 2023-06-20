# coding=utf8

import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from resnet import resnet50


def get_dataset(in_dir):
    # Initialize transformations for data augmentation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=45),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # Load the ImageNet Object Localization Challenge dataset
    ds = torchvision.datasets.ImageFolder(
        # root='/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train',
        root=in_dir,
        transform=transform
    )
    return ds


def get_dataloader(ds, args):
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_work)
    return dl


def get_model(args):
    # model = torchvision.models.resnet50(pretrained=True)
    model = resnet50()
    key_arr = ['conv', 'downsample.0', 'fc'] 
    cnt = 0
    for name, _module in model.named_modules():
        if _is_trace_module(key_arr, name):
            print(name)
            print(_module)
            cnt += 1
    print("num_layers: ", cnt)
    return model


def _is_trace_module(key_arr, name):
    for key in key_arr:
        if key in name.lower():
            return True
    return False


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='input dir of the image')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_work', type=int,  default=1)
    args = parser.parse_args()
    return args


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args)
    model = model.to(device)
    ds = get_dataset(args.input)
    train_loader = get_dataloader(ds, args)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # Backward pass
            loss.backward()
            optimizer.step()
        # Print the loss for every epoch
        print(f'Epoch {epoch+1}/{args.epochs}, Loss: {loss.item():.4f}')
    print(f'Finished Training, Loss: {loss.item():.4f}')


if __name__ == '__main__':
    args = get_args()
    train(args)
