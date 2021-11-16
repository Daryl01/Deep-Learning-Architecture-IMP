import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator, initialize_weights



# Hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNEL_IMG = 1
# CHANNEL_IMG = 3 # if using the celeb dataset
Z_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN =64
NOISE_DIM = 256
path2data = "/media/daryl-loyck/part1/data/pytcvcookbook/data"

transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(), 
    transforms.Normalize(
        [0.5 for _ in range(CHANNEL_IMG)], [0.5 for _ in range(CHANNEL_IMG)]
    ),
])

dataset = datasets.MNIST(root=path2data, train=True, transform=transforms, download=False)
# dataset = datasets.ImageFolder(root='pathtoimageceleb', transform=transforms)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# initialize discriminator and Generator
disc = Discriminator(CHANNEL_IMG, FEATURES_DISC).to(device)
gen = Generator(Z_DIM, CHANNEL_IMG, FEATURES_GEN).to(device)
initialize_weights(gen)
initialize_weights(disc)

# optim
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

# train state
criterion = nn.BCELoss()

fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
writer_fake = SummaryWriter(f"logs/DCGAN_MNIST/fake")
writer_real = SummaryWriter(f"logs/DCGAN_MNIST/real")
step = 0

gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        # batch_size = real.shape[0]
        noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
        fake = gen(noise)

        ### train the discriminator max log(D(real)) + log(1-D(G(z)))
        
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real)*0.9)
        disc_fake = disc(fake).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.ones_like(disc_fake)*0.1)
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        opt_disc.step()


        ### Train the Generetor min log(1 - D(G(Z))) <--> max log(D(G(Z)))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx % 100 == 0:
            print (
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch [{batch_idx}/{len(loader)}] Loss D : {loss_disc:.4f}, Loss G : {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)

                writer_fake.add_image("MNIST fake Images", img_grid_fake, global_step=step)
                writer_real.add_image("MNIST reak Images", img_grid_real, global_step=step)

                step += 1

