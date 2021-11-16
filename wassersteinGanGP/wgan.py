import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator, initialize_weights
from utils import gradient_penalty



# Hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNEL_IMG = 3
# CHANNEL_IMG = 3 # if using the celeb dataset
Z_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN =64
CRITIC_ITERATIONS =5
LAMBDA_GP = 10

path2data = "/media/daryl-loyck/part1/data/pytcvcookbook/data"
path2celebdata = "/media/daryl-loyck/part1/data/dl_tuto_data/images1"

transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(), 
    transforms.Normalize(
        [0.5 for _ in range(CHANNEL_IMG)], [0.5 for _ in range(CHANNEL_IMG)]
    ),
])

dataset = datasets.MNIST(root=path2data, train=True, transform=transforms, download=False)
# dataset = datasets.ImageFolder(root=path2celebdata, transform=transforms)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# initialize discriminator and Generator
critic = Discriminator(CHANNEL_IMG, FEATURES_DISC).to(device)
gen = Generator(Z_DIM, CHANNEL_IMG, FEATURES_GEN).to(device)
initialize_weights(gen)
initialize_weights(critic)

# optim
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))



fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")
step = 0

gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            gp = gradient_penalty(critic, real, fake, device=device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) +
                LAMBDA_GP*gp
            )
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()


        ### Train the Generetor min log(1 - D(G(Z))) <--> max log(D(G(Z)))
        output = critic(fake).reshape(-1)
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()


        if batch_idx % 100 == 0:
            print (
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch [{batch_idx}/{len(loader)}] Loss D : {loss_critic:.4f}, Loss G : {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)

                writer_fake.add_image("MNIST fake Images", img_grid_fake, global_step=step)
                writer_real.add_image("MNIST reak Images", img_grid_real, global_step=step)

                step += 1

