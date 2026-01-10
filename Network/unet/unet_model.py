""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = (DoubleConv(in_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        # Removed down3 layer
        factor = 2 if bilinear else 1
        self.down3 = (Down(256, 512 // factor))
        
        self.up1 = (Up(512, 256 // factor, bilinear))
        self.up2 = (Up(256, 128 // factor, bilinear))
        self.up3 = (Up(128, 64, bilinear))
        # Removed up4 layer
        self.outc = (OutConv(64, out_channels))

        self.att_1 = AttentionGroup(64)
        self.att_2 = AttentionGroup(128)
        self.att_3 = AttentionGroup(256)
        self.att_4 = AttentionGroup(512)

        self.att_5 = AttentionGroup(256)
        self.att_6 = AttentionGroup(128)
        self.att_7 = AttentionGroup(64)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.att_1(x1)

        x2 = self.down1(x1)
        x2 = self.att_2(x2)

        x3 = self.down2(x2)
        x3 = self.att_3(x3)

        x4 = self.down3(x3)
        x4 = self.att_4(x4)
        
        x = self.up1(x4, x3)
        x = self.att_5(x)

        x = self.up2(x, x2)
        x = self.att_6(x)

        x = self.up3(x, x1)
        x = self.att_7(x)
        
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.outc = torch.utils.checkpoint(self.outc)