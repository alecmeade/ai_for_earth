import torch
import torch.nn as nn
import torch.nn.functional as F

from enum import Enum

class UNet(nn.Module):
    """UNet specified in https://arxiv.org/abs/1505.04597, structure is modified from
    a combination of https://github.com/milesial/Pytorch-UNet 
    and https://github.com/usuyama/pytorch-unet.
    
    
    Args:
        input_channels: Number of input channels to the network.
        n_classes: Number of output channels to the network.
        depth: The depth of the network.
        start_channels: The number of channels to use at the first layer of the UNet
            and scale from there.
        scale_factor: The scale factor to use when increasing and decreasing the
            number of channels in the contraction and expansion stages of the network
            respectively.
    """
    def __init__(self, 
                 in_channels:int,
                 n_classes: int, 
                 depth: int, 
                 start_channels: int = 64, 
                 scale_factor: int = 2):
        super().__init__()
        # Stores the contraction and expansion layers in the UNet.
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()

        prev_layer_channels = in_channels
        out_channels = start_channels
        self.start_layer = DoubleConvBlock(prev_layer_channels, out_channels,
                kernel_size = 3, padding = 1, has_relu = True, has_batch_norm = True)
    
        for i in range(depth):
            # Create contraction layers that increase the input channels by the provided
            # scale factor.
            prev_layer_channels = out_channels
            out_channels = prev_layer_channels * scale_factor
            self.down_layers.append(DownLayer(prev_layer_channels,
                                              out_channels, 
                                              scale_factor = scale_factor))
        
         
        for i in range(depth):
            # Create contraction layers that reduce the input channels by the provided
            # scale factor.
            prev_layer_channels = out_channels
            out_channels = int(prev_layer_channels / scale_factor)
            self.up_layers.append(UpLayer(prev_layer_channels,
                                          out_channels, 
                                          scale_factor = scale_factor))

        # Final layer in the network performing a 1x1 convolution to match the number of
        # output classes.
        prev_layer_channels = out_channels
        out_channels = int(prev_layer_channels / scale_factor)
        self.conv1d = nn.Conv2d(prev_layer_channels, n_classes, kernel_size=1)
        
    def forward(self, x):
        """Forward pass for the UNet performing both contraction, contraction and skip connections."""
        out = x 
        down_outs = []
        out = self.start_layer(x)
        down_outs.append(out)

        # Pass input through contraction layers and store the output for use with skip connections.
        for down in self.down_layers:
            out = down(out)
            down_outs.append(out)
    
        # Pass input through the expansion layers and add in the corresponding contraction layer output.
        for i, up in enumerate(self.up_layers):
            down_out = down_outs[-(i + 2)]
            out = up(out, down_out)

        return self.conv1d(out)

        
class DoubleConvBlock(nn.Module):
    """A module representing a two repeated convolution steps..
    
    This step is composed of a double convolution followed by max pooling.

    Args:
        in_channels: The number of input channels to the layer.
        out_channels: The number of output channels to the layer.
        kernel_size: The kernel size of the double convolution.
        padding: The number of pixels to add as padding.
        has_relu: Whether the convolution step should apply a relu activation.
        has_batch_norm: Whether the convolution step should apply a batch normalization.
    """
    def __init__(self, 
                 in_channels:int, 
                 out_channels:int, 
                 kernel_size: int,
                 padding: int,
                 has_relu: bool, 
                 has_batch_norm: bool):
        
        super().__init__()
        self.conv_layers = nn.ModuleList()
        
        def addConvLayer(in_c, out_c):
            """Creates a single convolutional layer."""
            self.conv_layers.append(nn.Conv2d(in_c, out_c, kernel_size, padding=padding))
            if has_relu:
                self.conv_layers.append(nn.ReLU())

            if has_batch_norm:
                self.conv_layers.append(nn.BatchNorm2d(out_c))
        
        # Create two convolutional layers with the first performing the necessary
        # reduction in channels.
        addConvLayer(in_channels, out_channels)
        addConvLayer(out_channels, out_channels)
        self.conv_layers = nn.Sequential(*self.conv_layers)
            
    def forward(self, x):
        """Forward pass for the double convolution layer."""
        return self.conv_layers(x)

class DownLayer(nn.Module):
    """A module representing a single contraction step in a UNet.
    
    This step is composed of a double convolution followed by max pooling.

    Args:
        in_channels: The number of input channels to the layer.
        out_channels: The number of output channels to the layer.
        kernel_size: The kernel size of the double convolution.
        padding: The number of pixels to add as padding.
        has_relu: Whether the convolution step should apply a relu activation.
        has_batch_norm: Whether the convolution step should apply a batch normalization.
        scale_factor: The factor by which to multiply in_channels by in each convolution and pooling step.
    """
    def __init__(self,
                 in_channels:int, 
                 out_channels:int, 
                 kernel_size: int = 3,
                 padding: int = 1,
                 has_relu: bool = True, 
                 has_batch_norm: bool = True,
                 scale_factor: int = 2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(scale_factor),
            DoubleConvBlock(in_channels, 
                            out_channels, 
                            kernel_size,
                            padding,
                            has_relu, 
                            has_batch_norm))

    def forward(self, x):
        """Forward pass for the Downlayer."""
        return self.layers(x)

    
class UpConvMode(Enum):
    """Class represting the possible modes for the expansion half of the UNet."""
    CONV_TRANSPOSE = 1
    UPSAMPLE = 2
    
class InvalidUpConvModeError(Exception):
    """Error raised when an invalid up convolution mode is selected for a UNet."""
    pass


class UpLayer(nn.Module):
    """A module representing a single expansion step in a UNet.
    
    This step is composed of a deconvolution or upsampling combined with a copying and
    contentation of outputs from the corresponding DownLayer which are then convolved
    together.

    Args:
        in_channels: The number of input channels to the layer.
        out_channels: The number of output channels to the layer.
        kernel_size: The kernel size of the deconvolution and subsequent convolution.
        padding: The number of pixels to add as padding.
        has_relu: Whether the convolution step should apply a relu activation.
        has_batch_norm: Whether the convolution step should apply a batch normalization.
        scale_factor: The factor by which to multiply in_channels by.
        stride: The stride of the ConvTranspose2d. This only takes effect if
            up_conv_mode == CONV_TRANSPOSE.
        up_conv_mode: The manner in which to perform the expansion step in the uplayer
            which can be accomplished by deconvolution or upsampling.
    """

    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int = 3,
                 padding: int = 1,
                 has_relu: bool = True, 
                 has_batch_norm: bool = True,
                 scale_factor: int = 2,
                 stride: int = 2,
                 up_conv_mode: UpConvMode = UpConvMode.CONV_TRANSPOSE):
        super().__init__()
        up_layers = []
    
        if up_conv_mode == UpConvMode.CONV_TRANSPOSE:
            # Use deconvolution to increase the features in the UNet.
            up_layers.append(nn.ConvTranspose2d(in_channels, out_channels, scale_factor, stride))
                          
        elif up_conv_mode == UpConvMode.UPSAMPLE:
            # Use upsampling followed by convolution to increase the features in the UNet.
            up_layers.append(nn.Upsample(mode = 'bilinear', scale_factor=scale_factor))
            up_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size))
                          
        else:
            # Only UpConvMode.CONV_TRANSPOSE and UpConvMode.UPSAMPLE are supported.
            raise InvalidUpConvModeError
        
        self.up_layers = nn.Sequential(*up_layers)
        
        # Create a convolution layer that will take as input the output of the `self.up_layers`
        # and the output from a skip level connection to the contraction side of the UNet.
        self.conv_layers = DoubleConvBlock(in_channels, 
                                           out_channels, 
                                           kernel_size,
                                           padding,
                                           has_relu, 
                                           has_batch_norm)

    def forward(self, up_x, down_x):
        """Forward pass for the UpLayer which combines both the contraction and expansion output."""

        up_out = self.up_layers(up_x)
        
        # Crop the output from the contraction side of the UNet to remove the borders
        # added by strided convolutions.
        h_up, w_up = up_out.size()[2:]
        h_down, w_down = down_x.size()[2:]
        diff_h = int((h_down - h_up) / 2)
        diff_w = int((w_down - w_up) / 2)
        crop_out = down_x[:, :, diff_h:(diff_h + h_up), diff_w:(diff_w + w_up)]
        # Concat the output of the expansion and contraction layers.
        concat_out = torch.cat([crop_out, up_out], dim = 1)


        return self.conv_layers(concat_out)
