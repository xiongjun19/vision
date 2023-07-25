# coding=utf8

import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
from resnet import resnet50
from timer import Timers
from torch import autocast
from torch.cuda.amp import GradScaler


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
    sampler = None
    if args.world_size > 1:
        sampler = torch.utils.data.DistributedSampler(ds, shuffle=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, 
            num_workers=args.num_work, pin_memory=True,
            sampler=sampler)
    return dl


def get_model(args):
    # model = torchvision.models.resnet50(pretrained=True)
    model = resnet50()
    return model


def register_forward_hooks(model, timer):
    key_arr = ['conv', 'downsample.0', 'fc']
    cnt = 0
    base_name = 'fwd'
    for name, _module in model.named_modules():
        if _is_trace_module(key_arr, name):
            time_key = f'{base_name}_{cnt}'
            cnt += 1
            _module.register_forward_pre_hook(partial(pre_fw_hook, timer, time_key))
            _module.register_forward_hook(partial(fw_hook, timer, time_key))
    print("cnt is ", cnt)
    print("complete register forward hooks")


def pre_fw_hook(timer, name, module, *argv):
    timer(name).start()


def fw_hook(timer, name, module, *argv):
    timer(name).stop()
    timer.log([name])


def bw_ig_hook(timer, name, *argv):
    prefix = 'ig_' + name
    timer.log(["backward"], reset=False, prefix=prefix)


def bw_wg_hook(timer, name, *argv):
    prefix = 'wg_' + name
    timer.log(["backward"], reset=False, prefix=prefix)


def register_backward_hooks(model, timer):
    key_arr = ['conv', 'downsample.0', 'fc']
    cnt = 0
    base_name = 'bwd'
    for name, _module in model.named_modules():
        if _is_trace_module(key_arr, name):
            time_key = f'{base_name}_{cnt}'
            cnt += 1
            _module.register_backward_hook(partial(bw_ig_hook, timer, time_key))
            _module.weight.register_hook(partial(bw_wg_hook, timer, time_key))
    print("cnt is ", cnt)
    print("complete register forward hooks")


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
    parser.add_argument('--num_work', type=int, default=1)
    parser.add_argument('--no_overlap', action='store_true')
    group = parser.add_argument_group(title='distributed')
    group.add_argument('--distributed-backend', default='nccl',
                       choices=['nccl', 'gloo'],
                       help='Which backend to use for distributed training.')
    group.add_argument('--DDP-impl', default='local',
                       choices=['local', 'torch'],
                       help='which DistributedDataParallel implementation '
                       'to use.')
    group.add_argument('--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher.')
    args = parser.parse_args()
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))
    return args


def _initialize_distributed(args):
    device_count = torch.cuda.device_count()
    if torch.distributed.is_initialized():
        if args.rank == 0:
            print('torch distributed is already initialized, '
                  'skipping initialization ...', flush=True)
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
    else:
        if args.rank == 0:
            print('> initializing torch distributed ...', flush=True)
        # Manually set the device ids.
        if device_count > 0:
            device = args.rank % device_count
            if args.local_rank is not None:
                assert args.local_rank == device, \
                    'expected local-rank to be the same as rank % device-count.'
            else:
                args.local_rank = device
            torch.cuda.set_device(device)
    # Call the init process
    torch.distributed.init_process_group(
        backend=args.distributed_backend,
        world_size=args.world_size,
        rank=args.rank)


def train(args):
    timer = Timers()
    _initialize_distributed(args)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args)
    if args.local_rank >= 0:
        model = model.cuda()
    model = model.to(memory_format=torch.channels_last)
    register_forward_hooks(model, timer)
    register_backward_hooks(model, timer)
    if torch.distributed.get_world_size() > 1:
        model = torch.nn.parallel.DistributedDataParallel(model)
    ds = get_dataset(args.input)
    train_loader = get_dataloader(ds, args)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scaler = GradScaler()
    tot_iters = 20
    iter_no = 0
    for epoch in range(args.epochs):
        if train_loader.sampler is not None:
            if args.world_size > 1:
                train_loader.sampler.set_epoch(epoch)
        for inputs, labels in train_loader:
            iter_str = f"iteration number:{iter_no}"
            if torch.distributed.is_initialized():
                if torch.distributed.get_rank() == (
                        torch.distributed.get_world_size() - 1):
                    print(iter_str, flush=True)
            else:
                print(iter_str, flush=True)

            # inputs = inputs.to(device)
            # labels = labels.to(device)
            inputs = inputs.cuda()
            labels = labels.cuda()
            inputs = inputs.to(memory_format=torch.channels_last, dtype=torch.float16) 
            # labels = labels.half()
            optimizer.zero_grad()
            # Forward pass
            if args.world_size > 1 and args.no_overlap:
                with model.no_sync():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    # Backward pass
                    timer('backward').start()
                    loss.backward()
            else:
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                # Backward pass
                timer('backward').start()
                # loss.backward()
                scaler.scale(loss).backward()

            if args.world_size > 1 and args.no_overlap:
                for param in model.parameters():
                    if param.requires_grad and param.grad is not None:
                        torch.distributed.all_reduce(param.grad)
            timer('backward').stop()
            timer.log(['backward'])
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()
            iter_no += 1
            if iter_no >= tot_iters:
                break

        # Print the loss for every epoch
        print(f'Epoch {epoch+1}/{args.epochs}, Loss: {loss.item():.4f}')
    print(f'Finished Training, Loss: {loss.item():.4f}')


if __name__ == '__main__':
    args = get_args()
    train(args)
