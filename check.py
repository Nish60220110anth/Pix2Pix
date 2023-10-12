from discriminator import Discriminator
from generator import Generator

from torch import randn


def check_discriminator():
    x = randn((1, 3, 256, 256))
    y = randn((1, 3, 256, 256))

    model = Discriminator(in_channels=3)
    output = model(x, y)

    print(output.shape)


def check_generator():
    x = randn((1, 3, 256, 256))
    model = Generator(3)

    out = model(x)
    print(out.shape)


if __name__ == "__main__":
    check_generator()
    check_discriminator()
