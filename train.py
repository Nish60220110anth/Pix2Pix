import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
import config
from dataset import MapDataset
from discriminator import Discriminator
from generator import Generator

torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.deterministic = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    disc = Discriminator(in_channels=3).to(device=device)
    gen = Generator(in_channels=3, start_feature=64).to(device=device)

    opt_dics = torch.optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = torch.optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    L1 = nn.L1Loss().to(device)
    bsc = nn.BCEWithLogitsLoss().to(device)

    train_dataset = MapDataset(root_dir=config.TRAIN_DIR)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    val_dataset = MapDataset(root_dir=config.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=config.NUM_WORKERS)

    for epoch in range(config.NUM_EPOCHS):
        loop = tqdm(train_loader, leave=True)
        for idx, (x, y) in enumerate(loop):
            x = x.to(device)
            y = y.to(device)

            # Train Discriminator
            with torch.cuda.amp.autocast():
                y_fake = gen(x)
                D_real = disc(x, y)
                D_fake = disc(x, y_fake.detach())
                D_real_loss = bsc(D_real, torch.ones_like(D_real))
                D_fake_loss = bsc(D_fake, torch.zeros_like(D_fake))
                D_loss = (D_real_loss + D_fake_loss) / 2

            opt_dics.zero_grad()
            D_loss.backward()
            opt_dics.step()

            # Train Generator
            with torch.cuda.amp.autocast():
                D_fake = disc(x, y_fake)
                G_fake_loss = bsc(D_fake, torch.ones_like(D_fake))
                L1_loss = L1(y_fake, y) * config.L1_LAMBDA
                G_loss = G_fake_loss + L1_loss

            opt_gen.zero_grad()
            G_loss.backward()
            opt_gen.step()

            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )
        
        if config.SAVE_MODEL and epoch % 5 == 0:
            torch.save(gen.state_dict(), f"saved_models/gen_{epoch}.pth")
            torch.save(disc.state_dict(), f"saved_models/disc_{epoch}.pth")

if __name__ == "__main__":
    main()