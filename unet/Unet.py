import torch 
import torch.nn as nn
from cbn3 import CBN


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=48, channels=512):
        super(UNet, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_block1 = UnetBlock(in_channels, out_channels) #512, 64
        self.down_block2 = UnetBlock(out_channels, out_channels*2) #256, 96
        self.down_block3 = UnetBlock(out_channels*2, out_channels*4) # 128, 192
        self.down_block4 = UnetBlock(out_channels*4, out_channels*8) # 64, 384
        self.down_block5 = UnetBlock(out_channels*8, channels) # 32, 512

        self.conv_down = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)


        # # upsampling
        # #1
        self.up_conv = UnetBlock(256, 512) # 32, 512
        self.up_trans_1 = nn.ConvTranspose2d(
            in_channels=512, 
            out_channels=384, 
            kernel_size=2, 
            stride=2
        )
        self.up_conv_1 = UnetBlock((384*2), 384)
        self.cbn1 = CBN(2, 192, 384) # 64, 384

        #2 
        self.up_trans_2 = nn.ConvTranspose2d(
            in_channels=384, 
            out_channels=192, 
            kernel_size=2, 
            stride=2
        ) # 128, 192
        self.up_conv_2 = UnetBlock(384, 192)
        self.cbn2 = CBN(2, 128, 192) # 128, 192

        #3
        self.up_trans_3 = nn.ConvTranspose2d(
            in_channels=192, 
            out_channels=96, 
            kernel_size=2, 
            stride=2
        )
        self.up_conv_3 =  UnetBlock(192, 96)
        self.cbn3 = CBN(2, 256, 96) # 256, 96

        #4
        self.up_trans_4 = nn.ConvTranspose2d(
            in_channels=96, 
            out_channels=48, 
            kernel_size=2, 
            stride=2
        )
        self.up_conv_4 = UnetBlock(96, 48)
        self.cbn4 = CBN(2, 96, 48) # 512, 48


        self.out = nn.Conv2d(
            in_channels=48, 
            out_channels=3, 
            kernel_size=3, padding=1)


    
    def forward(self, image, mlp_params):

        #encoder
        out1 = self.down_block1(image) # copy and cropped

        out2 = self.down_block2(out1) # copy and cropped
        out2 = self.maxpool(out2)

        out3 = self.down_block3(out2) # copy and cropped
        out3 = self.maxpool(out3)

        out4 = self.down_block4(out3) # copy and cropped
        out4 = self.maxpool(out4)

        out5 = self.down_block5(out4)
        out5 = self.maxpool(out5)

        out_encoder = self.conv_down(out5)


        # Decoder
        #1
        x = self.up_conv(out_encoder)
        x = self.up_trans_1(x)
        x = self.up_conv_1 (torch.cat([out4, x], 1))
        x = self.cbn1(x, mlp_params)
        
        #2
        x2 = self.up_trans_2(x) 
        x2 = self.up_conv_2(torch.cat([out3, x2], 1))
        x2 = self.cbn2(x2, mlp_params)

        # #3
        x3 = self.up_trans_3(x2)
        x3 = self.up_conv_3(torch.cat([out2, x3], 1))
        x3 = self.cbn3(x3, mlp_params)

        #4
        x4 = self.up_trans_4(x3)
        x4 = self.up_conv_4(torch.cat([out1, x4], 1))
        x4 = self.cbn4(x4, mlp_params)
        
        out = self.out(x4)
        print(out.size())
        return out


def crop_img(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta  = delta // 2
    return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]


class UnetBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, conv_channels, kernel_size=3, padding=1, bias=True,
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            conv_channels, conv_channels, kernel_size=3, padding=1, bias=True,
        )
        self.relu2 = nn.ReLU(inplace=True)


    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)

        return block_out




if __name__ == "__main__":
    image = torch.rand((1, 3, 512, 512))
    # print(torch.mean(image))
    # print(torch.var(image))
    model = UNet()
    batch_size = image.size()[0]
    # MLP input (batch_size, number of param cnb need to learn)
    cat = torch.zeroes([4, 2])
    for i in range(batch_size//2):
        cat[i, 0] = 1
    for i in range(batch_size//2, batch_size):
        cat[i, 1] = 1
    #for i in range()
    # print(model(image, cat))