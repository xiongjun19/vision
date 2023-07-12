# coding=utf8

from torchstat import stat
from resnet import resnet50


def main():
    model = resnet50()
    stat(model, (3, 224, 224))


if __name__ == '__main__':
    main()
