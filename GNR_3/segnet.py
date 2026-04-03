import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_CLASSES = 11


def conv_block(in_channels, out_channels):
    # Simple encoder/decoder block with two convolutions.
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class SharedEncoder(nn.Module):
    def __init__(self, input_channels, encoder_channels):
        super().__init__()
        c1, c2, c3 = encoder_channels

        self.enc_conv1 = conv_block(input_channels, c1)
        self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)

        self.enc_conv2 = conv_block(c1, c2)
        self.pool2 = nn.MaxPool2d(2, 2, return_indices=True)

        self.enc_conv3 = conv_block(c2, c3)
        self.pool3 = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, x):
        x1 = self.enc_conv1(x)
        x1_size = x1.size()
        x1_pooled, ind1 = self.pool1(x1)

        x2 = self.enc_conv2(x1_pooled)
        x2_size = x2.size()
        x2_pooled, ind2 = self.pool2(x2)

        x3 = self.enc_conv3(x2_pooled)
        x3_size = x3.size()
        x3_pooled, ind3 = self.pool3(x3)

        return {
            "skip1": x1,
            "skip2": x2,
            "skip3": x3,
            "x": x3_pooled,
            "size1": x1_size,
            "size2": x2_size,
            "size3": x3_size,
            "ind1": ind1,
            "ind2": ind2,
            "ind3": ind3,
        }


class SegNetBasic(nn.Module):
    def __init__(self, input_channels=3, encoder_channels=(64, 128, 256), num_classes=NUM_CLASSES):
        super().__init__()
        c1, c2, c3 = encoder_channels

        self.encoder = SharedEncoder(input_channels, encoder_channels)

        self.unpool3 = nn.MaxUnpool2d(2, 2)
        self.dec_conv3 = conv_block(c3, c2)

        self.unpool2 = nn.MaxUnpool2d(2, 2)
        self.dec_conv2 = conv_block(c2, c1)

        self.unpool1 = nn.MaxUnpool2d(2, 2)
        self.dec_conv1 = conv_block(c1, c1)

        self.final = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x):
        encoded = self.encoder(x)

        x = self.unpool3(encoded["x"], encoded["ind3"], output_size=encoded["size3"])
        x = self.dec_conv3(x)

        x = self.unpool2(x, encoded["ind2"], output_size=encoded["size2"])
        x = self.dec_conv2(x)

        x = self.unpool1(x, encoded["ind1"], output_size=encoded["size1"])
        x = self.dec_conv1(x)

        return self.final(x)


class SegNetEncoderAddition(nn.Module):
    def __init__(self, input_channels=3, encoder_channels=(64, 128, 256), num_classes=NUM_CLASSES):
        super().__init__()
        c1, c2, c3 = encoder_channels

        self.encoder = SharedEncoder(input_channels, encoder_channels)

        self.unpool3 = nn.MaxUnpool2d(2, 2)
        self.dec_conv3 = conv_block(c3, c2)

        self.unpool2 = nn.MaxUnpool2d(2, 2)
        self.dec_conv2 = conv_block(c2, c1)

        self.unpool1 = nn.MaxUnpool2d(2, 2)
        self.dec_conv1 = conv_block(c1, c1)

        self.final = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x):
        encoded = self.encoder(x)

        x = self.unpool3(encoded["x"], encoded["ind3"], output_size=encoded["size3"])
        x = x + encoded["skip3"]
        x = self.dec_conv3(x)

        x = self.unpool2(x, encoded["ind2"], output_size=encoded["size2"])
        x = x + encoded["skip2"]
        x = self.dec_conv2(x)

        x = self.unpool1(x, encoded["ind1"], output_size=encoded["size1"])
        x = x + encoded["skip1"]
        x = self.dec_conv1(x)

        return self.final(x)


class SegNetSkip(nn.Module):
    def __init__(self, input_channels=3, encoder_channels=(64, 128, 256), num_classes=NUM_CLASSES):
        super().__init__()
        c1, c2, c3 = encoder_channels

        self.encoder = SharedEncoder(input_channels, encoder_channels)

        self.unpool3 = nn.MaxUnpool2d(2, 2)
        self.dec_conv3 = conv_block(c3 + c3, c2)

        self.unpool2 = nn.MaxUnpool2d(2, 2)
        self.dec_conv2 = conv_block(c2 + c2, c1)

        self.unpool1 = nn.MaxUnpool2d(2, 2)
        self.dec_conv1 = conv_block(c1 + c1, c1)

        self.final = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x):
        encoded = self.encoder(x)

        x = self.unpool3(encoded["x"], encoded["ind3"], output_size=encoded["size3"])
        x = torch.cat([x, encoded["skip3"]], dim=1)
        x = self.dec_conv3(x)

        x = self.unpool2(x, encoded["ind2"], output_size=encoded["size2"])
        x = torch.cat([x, encoded["skip2"]], dim=1)
        x = self.dec_conv2(x)

        x = self.unpool1(x, encoded["ind1"], output_size=encoded["size1"])
        x = torch.cat([x, encoded["skip1"]], dim=1)
        x = self.dec_conv1(x)

        return self.final(x)


class BilinearInterpolationModel(nn.Module):
    def __init__(self, input_channels=3, encoder_channels=(64, 128, 256), num_classes=NUM_CLASSES):
        super().__init__()
        self.encoder = SharedEncoder(input_channels, encoder_channels)
        self.classifier = nn.Conv2d(encoder_channels[-1], num_classes, kernel_size=1)

    def forward(self, x):
        encoded = self.encoder(x)
        x = self.classifier(encoded["x"])
        return F.interpolate(x, size=encoded["size1"][2:], mode="bilinear", align_corners=False)


class FCNBasic(nn.Module):
    def __init__(self, input_channels=3, encoder_channels=(64, 128, 256), num_classes=NUM_CLASSES):
        super().__init__()
        c1, c2, c3 = encoder_channels
        self.encoder = SharedEncoder(input_channels, encoder_channels)

        # Reduce all skip maps to num_classes channels before adding them.
        self.score3 = nn.Conv2d(c3, num_classes, kernel_size=1)
        self.score2 = nn.Conv2d(c3, num_classes, kernel_size=1)
        self.score1 = nn.Conv2d(c2, num_classes, kernel_size=1)

        self.up3 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=2, stride=2)

    def forward(self, x):
        encoded = self.encoder(x)

        x = self.score3(encoded["x"])
        x = self.up3(x) + self.score2(encoded["skip3"])
        x = self.up2(x) + self.score1(encoded["skip2"])
        x = self.up1(x)

        return x


class FCNBasicNoDimReduction(nn.Module):
    def __init__(self, input_channels=3, encoder_channels=(64, 128, 256), num_classes=NUM_CLASSES):
        super().__init__()
        c1, c2, c3 = encoder_channels
        self.encoder = SharedEncoder(input_channels, encoder_channels)

        self.up3 = nn.ConvTranspose2d(c3, c3, kernel_size=2, stride=2)
        self.dec3 = conv_block(c3, c2)

        self.up2 = nn.ConvTranspose2d(c2, c2, kernel_size=2, stride=2)
        self.dec2 = conv_block(c2, c1)

        self.up1 = nn.ConvTranspose2d(c1, c1, kernel_size=2, stride=2)
        self.dec1 = conv_block(c1, c1)

        self.final = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x):
        encoded = self.encoder(x)

        x = self.up3(encoded["x"]) + encoded["skip3"]
        x = self.dec3(x)

        x = self.up2(x) + encoded["skip2"]
        x = self.dec2(x)

        x = self.up1(x) + encoded["skip1"]
        x = self.dec1(x)

        return self.final(x)


VARIANT_CONFIGS = {
    "bilinear_interpolation": {
        "builder": BilinearInterpolationModel,
        "input_channels": 3,
        "encoder_channels": (64, 128, 256),
        "num_classes": NUM_CLASSES,
        "description": "Fixed bilinear upsampling baseline.",
    },
    "segnet_basic": {
        "builder": SegNetBasic,
        "input_channels": 3,
        "encoder_channels": (64, 128, 256),
        "num_classes": NUM_CLASSES,
        "description": "Basic SegNet with max-unpooling decoder.",
    },
    "segnet_basic_encoder_addition": {
        "builder": SegNetEncoderAddition,
        "input_channels": 3,
        "encoder_channels": (64, 128, 256),
        "num_classes": NUM_CLASSES,
        "description": "SegNet variant with encoder feature addition.",
    },
    "fcn_basic": {
        "builder": FCNBasic,
        "input_channels": 3,
        "encoder_channels": (64, 128, 256),
        "num_classes": NUM_CLASSES,
        "description": "Basic FCN-style learned upsampling model.",
    },
    "fcn_basic_no_dim_reduction": {
        "builder": FCNBasicNoDimReduction,
        "input_channels": 3,
        "encoder_channels": (64, 128, 256),
        "num_classes": NUM_CLASSES,
        "description": "FCN variant without dimensionality reduction.",
    },
    "segnet_skip": {
        "builder": SegNetSkip,
        "input_channels": 3,
        "encoder_channels": (64, 128, 256),
        "num_classes": NUM_CLASSES,
        "description": "Extra skip-connection improvement over basic SegNet.",
    },
}


def build_model(variant_name):
    if variant_name not in VARIANT_CONFIGS:
        valid = ", ".join(sorted(VARIANT_CONFIGS))
        raise ValueError(f"Unknown variant '{variant_name}'. Valid options: {valid}")

    config = VARIANT_CONFIGS[variant_name]
    builder = config["builder"]
    return builder(
        input_channels=config["input_channels"],
        encoder_channels=config["encoder_channels"],
        num_classes=config["num_classes"],
    )
