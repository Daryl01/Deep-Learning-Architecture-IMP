import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from model_utils import Discriminator, Generator

# Hyperparameters

batch_size = 64
image_size = 64
channels_img = 1
channels_noise = 256
num_epochs = 10

features_d = 16
features_g = 16
path2data = "/media/daryl-loyck/part1/data/pytcvcookbook/data"

device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4 # best lr rate for adam 3e-4

my_transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))
])


dataset = datasets.MNIST(root=path2data, train=True, transform=my_transforms, download=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# initialize discriminator and Generator
netD = Discriminator(channels_img, features_d).to(device)
netG = Generator(channels_noise, channels_img, features_g).to(device)

# optim
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

# train state
criterion = nn.BCELoss()

real_label = 1
fake_label = 0

# fixe noise to observe the same data as they progress in tensorboard
fixed_noise = torch.randn((batch_size, channels_noise, 1, 1)).to(device)

writer_fake = SummaryWriter(f"runs/DCGAN_MNIST/test_fake")
writer_real = SummaryWriter(f"runs/DCGAN_MNIST/test_real")
step = 0

print('start training')

for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(dataloader):
        if batch_idx == 0:
            print(data.shape)
            
            print
        data = data.to(device)
        batch_size = data.shape[0]

        ### train the discriminator max log(D(real)) + log(1-D(G(z)))
        netD.zero_grad()
        label = (torch.ones(batch_size)*0.9).to(device)
        output = netD(data).reshape(-1)
        lossD_real = criterion(output, label)
        D_x = output.mean().item()

        noise = torch.randn(batch_size, channels_noise, 1, 1).to(device)
        fake = netG(noise)
        label = (torch.ones(batch_size)*0.1).to(device)

        output = netD(fake.detach()).reshape(-1)
        lossD_fake = criterion(output, label)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()


        ### Train the Generetor min log(1 - D(G(Z))) <--> max log(D(G(Z)))
        netG.zero_grad()
        label = torch.ones(batch_size).to(device)
        output = netD(fake).reshape(-1)
        lossG = criterion(output, label)
        lossG.backward()
        optimizerG.step()

        if batch_idx == 0:
            print (
                f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(dataloader)}] Loss D : {lossD:.4f}, Loss G : {lossG:.4f} D(x): {D_x:.4f}"
            )

            with torch.no_grad():
                fake = netG(fixed_noise)
                # data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(data[:32], normalize=True)

                writer_fake.add_image("MNIST fake Images", img_grid_fake, global_step=step)
                writer_real.add_image("MNIST reak Images", img_grid_real, global_step=step)

                step += 1
