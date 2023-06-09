import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Conv2dReLU(layers.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):
        super(Conv2dReLU, self).__init__()

        self.conv = layers.Conv2D(
            out_channels,
            kernel_size,
            strides=stride,
            padding=padding,
            use_bias=not use_batchnorm,
        )
        self.relu = layers.ReLU()

        if use_batchnorm:
            self.bn = layers.BatchNormalization()
        else:
            self.bn = layers.Identity()

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SCSEModule(layers.Layer):
    def __init__(self, in_channels, reduction=16):
        super(SCSEModule, self).__init__()

        self.cSE = keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Conv2D(in_channels // reduction, 1),
            layers.ReLU(),
            layers.Conv2D(in_channels, 1),
            layers.Activation('sigmoid')
        ])
        self.sSE = keras.Sequential([
            layers.Conv2D(1, 1),
            layers.Activation('sigmoid')
        ])

    def call(self, inputs):
        return inputs * self.cSE(inputs) + inputs * self.sSE(inputs)


class ArgMax(layers.Layer):
    def __init__(self, dim=None):
        super(ArgMax, self).__init__()
        self.dim = dim

    def call(self, inputs):
        return tf.argmax(inputs, axis=self.dim)


class Clamp(layers.Layer):
    def __init__(self, min=0, max=1):
        super(Clamp, self).__init__()
        self.min, self.max = min, max

    def call(self, inputs):
        return tf.clip_by_value(inputs, self.min, self.max)


class Activation(layers.Layer):
    def __init__(self, name, **params):
        super(Activation, self).__init__()

        if name is None or name == "identity":
            self.activation = layers.Identity()
        elif name == "sigmoid":
            self.activation = layers.Activation('sigmoid')
        elif name == "softmax2d":
            self.activation = layers.Softmax(axis=1)
        elif name == "softmax":
            self.activation = layers.Softmax()
        elif name == "logsoftmax":
            self.activation = layers.Activation('log_softmax')
        elif name == "tanh":
            self.activation = layers.Activation('tanh')
        elif name == "argmax":
            self.activation = ArgMax()
        elif name == "argmax2d":
            self.activation = ArgMax(dim=1)
        elif name == "clamp":
            self.activation = Clamp()
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError(
                f"Activation should be callable/sigmoid/softmax/logsoftmax/tanh/"
                f"argmax/argmax2d/clamp/None; got {name}"
            )

    def call(self, inputs):
        return self.activation(inputs)


class Attention(layers.Layer):
    def __init__(self, name, **params):
        super(Attention, self).__init__()

        if name is None:
            self.attention = layers.Identity()
        elif name == "scse":
            self.attention = SCSEModule(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def call(self, inputs):
        return self.attention(inputs)


class DecoderBlock(layers.Layer):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super(DecoderBlock, self).__init__()

        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = Attention(attention_type, in_channels=out_channels)

    def call(self, inputs, skip=None):
        x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(inputs)
        if skip is not None:
            x = tf.keras.layers.Concatenate(axis=-1)([x, skip])
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(layers.Layer):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super(CenterBlock, self).__init__()

        self.conv1 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x


class UnetPlusPlusDecoder(tf.keras.Model):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
    ):
        super(UnetPlusPlusDecoder, self).__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        self.in_channels = [head_channels] + list(decoder_channels[:-1])
        self.skip_channels = list(encoder_channels[1:]) + [0]
        self.out_channels = decoder_channels

        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = layers.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)

        self.blocks = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(layer_idx + 1):
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1)
                    out_ch = self.out_channels[layer_idx]
                else:
                    out_ch = self.skip_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1 - depth_idx)
                    in_ch = self.skip_channels[layer_idx - 1]
                self.blocks[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
        self.blocks[f"x_{0}_{len(self.in_channels)-1}"] = DecoderBlock(
            self.in_channels[-1], 0, self.out_channels[-1], **kwargs
        )
        self.depth = len(self.in_channels) - 1

    def call(self, *features):
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        # start building dense connections
        dense_x = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(self.depth - layer_idx):
                if layer_idx == 0:
                    output = self.blocks[f"x_{depth_idx}_{depth_idx}"](features[depth_idx], features[depth_idx + 1])
                    dense_x[f"x_{depth_idx}_{depth_idx}"] = output
                else:
                    dense_l_i = depth_idx + layer_idx
                    cat_features = [
                        dense_x[f"x_{idx}_{dense_l_i}"]
                        for idx in range(depth_idx + 1, dense_l_i + 1)
                    ]
                    cat_features = tf.keras.layers.Concatenate(axis=-1)(cat_features + [features[dense_l_i + 1]])
                    dense_x[f"x_{depth_idx}_{dense_l_i}"] = self.blocks[f"x_{depth_idx}_{dense_l_i}"](
                        dense_x[f"x_{depth_idx}_{dense_l_i-1}"], cat_features
                    )
        dense_x[f"x_{0}_{self.depth}"] = self.blocks[f"x_{0}_{self.depth}"](dense_x[f"x_{0}_{self.depth-1}"])
        return dense_x[f"x_{0}_{self.depth}"]
    
from typing import Optional, Union, List
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Conv2D
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, UpSampling2D, Concatenate

class UnetPlusPlusModel(tf.keras.Model):
    def __init__(
        self,
        encoder_name = "resnet50",
        encoder_depth = 5,
        encoder_weights = "imagenet",
        decoder_use_batchnorm = True,
        decoder_channels = [256, 128, 64, 32, 16],
        decoder_attention_type = "scse",
        in_channels = 3,
        classes = 3,
        activation = "sigmoid",
        aux_params = None,
    ):
        super(UnetPlusPlusModel, self).__init__()

        self.encoder = ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=(256, 256, 3),
        )

        self.decoder = UnetPlusPlusDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "unetplusplus-{}".format(encoder_name)
        self.initialize()

