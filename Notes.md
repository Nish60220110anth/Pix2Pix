# Notes

1. `Bias` = False when using Batch Normalization
2. Use `LeakyReLU` instead of `ReLU` for better results (in most cases) in discriminator
3. Initial layer in discriminator in pix2pix is very diff from other's (based on the paper)

## Doubts?

1. Why I get nan loss when I train the model after few epochs? (even when I use low learning rate)