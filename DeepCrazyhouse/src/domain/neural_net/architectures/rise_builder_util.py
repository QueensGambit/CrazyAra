"""
@file: rise_builder_util.py
Created on 26.09.18
@project: crazy_ara_refactor
@author: queensgambit

Please describe what the content of this file is about
"""

from mxnet.gluon.nn import HybridSequential, Conv2D, BatchNorm, Dense, AvgPool2D
from mxnet.gluon import HybridBlock
from mxnet.gluon.contrib.nn import HybridConcurrent
from DeepCrazyhouse.src.domain.neural_net.architectures.builder_util import get_act, get_pool


class _SqueezeExcitation(HybridBlock):
    def __init__(self, name, nb_act_maps, ratio=16, act_type="relu"):

        super(_SqueezeExcitation, self).__init__(prefix=name)

        self.nb_act_maps = nb_act_maps
        self.body = HybridSequential(prefix="")

        nb_units_hidden = nb_act_maps // ratio
        with self.name_scope():
            self.body.add(AvgPool2D(pool_size=8))
            self.body.add(Dense(nb_units_hidden))
            self.body.add(get_act(act_type))
            self.body.add(Dense(nb_act_maps))
            self.body.add(get_act("sigmoid"))

    def hybrid_forward(self, F, x):
        """
        Compute forward pass

        :param F: Handle
        :param x: Input data to the block
        :return: Activation maps of the block
        """
        feature_scaling = self.body(x)
        out = F.broadcast_mul(x, F.reshape(data=feature_scaling, shape=(-1, self.nb_act_maps, 1, 1)))
        return out


class _InceptionResnetBlock(HybridBlock):
    def __init__(self, name, ch, res_scale_fac=0.2, act_type="relu", bn_mom=0.9, use_se=True, shortcut=True):
        super(_InceptionResnetBlock, self).__init__(prefix=name)

        self.shortcut = shortcut
        self.body = None
        self.bn0 = None
        self.act0 = None
        self.se0 = None
        self.block_name = name
        self.res_scale_fac = res_scale_fac
        self.use_se = use_se

        self.bn0 = BatchNorm(momentum=bn_mom, prefix="%s_bn0" % name, in_channels=ch)
        self.act0 = get_act(act_type, prefix="%s_%s0" % (name, act_type))

        if use_se is True:
            self.se0 = _SqueezeExcitation("%s_se0" % name, ch, 16, act_type)

    def hybrid_forward(self, F, x):
        """
        Compute forward pass

        :param F: Handle
        :param x: Input data to the block
        :return: Activation maps of the block
        """

        shortcut = x
        out = self.body(x)

        if self.shortcut is True:
            # scale down the output of the residual block activations to stabilize training
            out = out.__mul__(self.res_scale_fac)
            # connect the shortcut with the residual activations
            out = F.broadcast_add(shortcut, out, name=self.block_name)

        # apply batchnormalization and activation
        out = self.bn0(out)
        out = self.act0(out)

        # apply squeeze excitation
        if self.use_se is True:
            out = self.se0(out)

        return out


class _UpsampleBlock(HybridBlock):
    def __init__(self, name, scale=2, sample_type="nearest"):
        super(HybridBlock, self).__init__(prefix=name)
        self.scale = scale
        self.sample_type = sample_type

    def hybrid_forward(self, F, x):
        """
        Compute forward pass

        :param F: Handle
        :param x: Input data to the block
        :return: Activation maps of the block
        """
        out = F.UpSampling(x, scale=self.scale, sample_type=self.sample_type)
        return out


class _RiseResidualBlock(HybridBlock):
    """
    Definition of a residual block without any pooling operation
    """

    def __init__(self, channels, bn_mom, act_type, unit_name, use_se=True, res_scale_fac=0.2):
        """

        :param channels: Number of channels used in the conv-operations
        :param bn_mom: Batch normalization momentum
        :param act_type: Activation function to use
        :param unit_name: Unit name of the residual block (only used for description (string))
        """
        super(_RiseResidualBlock, self).__init__(unit_name)
        self.act_type = act_type
        self.unit_name = unit_name
        self.res_scale_fac = res_scale_fac

        self.use_se = use_se

        # branch 0
        self.body = HybridSequential()
        self.body.add(
            Conv2D(channels=channels, kernel_size=(3, 3), padding=(1, 1), use_bias=False, prefix="%s_conv0" % unit_name)
        )
        self.body.add(BatchNorm(momentum=bn_mom, prefix="%s_bn0" % self.unit_name))
        self.body.add(get_act(act_type, prefix="%s_%s0" % (unit_name, act_type)))

        self.body.add(
            Conv2D(channels=channels, kernel_size=(3, 3), padding=(1, 1), use_bias=False, prefix="%s_conv1" % unit_name)
        )
        self.body.add(BatchNorm(momentum=bn_mom, prefix="%s_bn1" % self.unit_name))

        self.act0 = get_act(act_type, prefix="%s_%s1" % (unit_name, act_type))

        if use_se is True:
            self.se0 = _SqueezeExcitation("%s_se0" % unit_name, channels, 16, act_type)

    def hybrid_forward(self, F, x):
        """
        Compute forward pass

        :param F: Handle
        :param x: Input data to the block
        :return: Activation maps of the block
        """

        shortcut = x
        out = self.body(x)

        # if self.shortcut is True:
        # scale down the output of the residual block activations to stabilize training
        if self.res_scale_fac is not None:
            out = out.__mul__(self.res_scale_fac)
        # connect the shortcut with the residual activations
        out = F.broadcast_add(shortcut, out, name=self.unit_name)

        # apply batchnormalization and activation
        # out = self.bn0(out)
        out = self.act0(out)

        # apply squeeze excitation
        if self.use_se is True:
            out = self.se0(out)

        return out


class _RiseBlockA(_InceptionResnetBlock):
    def __init__(self, name, in_ch, ch, res_scale_fac, act_type, bn_mom, use_se, shortcut, pool_type):
        """


        IN 8x8: 256 TOTAL

        BRANCH 0

        	b_0         b_0        b_0_0
        -> 24 Conv 1x1 -> POOL2D 2x2 -> 32 Conv 3x3

        -> 24-> 32

        BRANCH 1

        	b_0         b_0        b_0_0
        -> 24 Conv 1x1 -> POOL2D 4x4 -> 32 Conv 2x2

        -> 24-> 32


        BRANCH 2
        -> 24 Conv 1x1 -> 32 Conv 3x3
        -> 24 -> 32

        BRANCH 3
        -> 24 Conv 1x1 -> 24 Conv 3x1
        	       -> 24 Conv 1x3
        -> 24 -> 48


        BRANCH 4
        -> 24 Conv 1x1 -> 24 Conv 5x1
        	       -> 24 Conv 1x5
        -> 24 -> 48


        BRANCH 5
        -> 24 Conv 1x1 -> 24 Conv 3x1 -> 32 Conv 1x3
        	       -> 24 Conv 1x3 -> 32 Conv 3x1
        -> 24 -> 64

        """

        super(_RiseBlockA, self).__init__(name, ch, res_scale_fac, act_type, bn_mom, use_se, shortcut)

        self.body = HybridSequential(prefix="")

        # entry point for all branches
        self.branches = HybridConcurrent(axis=1, prefix="")

        ch_0_0 = 42
        ch_0_1 = 64

        ch_1_0 = 32
        ch_1_1 = 32

        ch_2_0 = 64
        ch_2_1 = 96
        ch_2_2 = 128

        with self.name_scope():
            # branch 0
            self.b_0 = HybridSequential()
            self.b_0.add(get_pool(pool_type, pool_size=(2, 2), strides=(2, 2)))
            self.b_0.add(Conv2D(channels=ch_0_0, kernel_size=(1, 1), in_channels=in_ch))
            self.b_0.add(get_act(act_type))
            self.b_0.add(Conv2D(channels=ch_0_1, kernel_size=(3, 3), padding=(1, 1), in_channels=ch_0_0, use_bias=True))
            self.b_0.add(get_act(act_type))
            self.b_0.add(_UpsampleBlock("upsample0", scale=2))

            # branch 1
            self.b_1 = HybridSequential()
            self.b_1.add(get_pool(pool_type, pool_size=(4, 4), strides=(4, 4)))
            self.b_1.add(Conv2D(channels=ch_1_0, kernel_size=(1, 1), in_channels=in_ch, use_bias=True))
            self.b_1.add(get_act(act_type))
            self.b_1.add(Conv2D(channels=ch_1_1, kernel_size=(3, 3), padding=(1, 1), in_channels=ch_1_0, use_bias=True))
            self.b_1.add(get_act(act_type))
            self.b_1.add(_UpsampleBlock("upsample0", scale=4))
            # branch 1
            # self.b_1 = HybridSequential()
            # self.b_1.add(Conv2D(channels=ch_1_0, kernel_size=(1, 1), in_channels=in_ch, use_bias=False))
            # self.b_1.add(get_act(act_type))

            # branch 2
            self.b_2 = HybridSequential()
            self.b_2.add(Conv2D(channels=ch_2_0, kernel_size=(1, 1), in_channels=in_ch))
            self.b_2.add(get_act(act_type))
            self.b_2.add(Conv2D(channels=ch_2_1, kernel_size=(3, 3), padding=(1, 1), in_channels=ch_2_0, use_bias=True))
            self.b_2.add(get_act(act_type))
            self.b_2.add(Conv2D(channels=ch_2_2, kernel_size=(3, 3), padding=(1, 1), in_channels=ch_2_1, use_bias=True))
            self.b_2.add(get_act(act_type))

            # concatenate all branches and add them to the body
            self.branches.add(self.b_0)
            self.branches.add(self.b_1)
            self.branches.add(self.b_2)
            # self.branches.add(self.b_3)
            # self.branches.add(self.b_4)

            self.body.add(self.branches)

            self.body.add(
                Conv2D(
                    channels=ch,
                    kernel_size=(1, 1),
                    prefix="%s_conv0" % name,
                    in_channels=ch_0_1 + ch_1_1 + ch_2_2,
                    use_bias=False,
                )
            )  # +ch_3_2+ch_4_2


class _RiseBlockB(_InceptionResnetBlock):
    def __init__(self, name, in_ch, ch, res_scale_fac, act_type, bn_mom, use_se, shortcut, pool_type):
        super(_RiseBlockB, self).__init__(name, ch, res_scale_fac, act_type, bn_mom, use_se, shortcut)

        self.body = HybridSequential(prefix="")

        # entry point for all branches
        self.branches = HybridConcurrent(axis=1, prefix="")

        ch_0_0 = 32
        ch_0_1 = 96
        ch_0_2 = 96

        ch_1_0 = 32
        ch_1_1 = 96
        ch_1_2 = 96

        ch_2_0 = 192

        with self.name_scope():
            # branch 0
            self.b_0 = HybridSequential()
            self.b_0.add(get_pool(pool_type, pool_size=(2, 2), strides=(2, 2)))
            self.b_0.add(Conv2D(channels=ch_0_0, kernel_size=(1, 1), in_channels=in_ch))
            self.b_0.add(get_act(act_type))
            self.b_0.add(
                Conv2D(channels=ch_0_1, kernel_size=(3, 1), padding=(0, 1), in_channels=ch_0_0, use_bias=False)
            )
            self.b_0.add(
                Conv2D(channels=ch_0_2, kernel_size=(1, 3), padding=(1, 0), in_channels=ch_0_1, use_bias=False)
            )
            self.b_0.add(_UpsampleBlock("upsample0", scale=2))

            # branch 1
            self.b_1 = HybridSequential()
            self.b_1.add(Conv2D(channels=ch_1_0, kernel_size=(1, 1), in_channels=in_ch))
            self.b_1.add(get_act(act_type))
            self.b_1.add(
                Conv2D(channels=ch_1_1, kernel_size=(3, 1), padding=(0, 1), in_channels=ch_1_0, use_bias=False)
            )
            self.b_1.add(
                Conv2D(channels=ch_1_2, kernel_size=(1, 3), padding=(1, 0), in_channels=ch_1_1, use_bias=False)
            )

            # branch 2
            self.b_2 = HybridSequential()
            self.b_2.add(Conv2D(channels=ch_2_0, kernel_size=(1, 1), in_channels=in_ch, use_bias=False))

            # concatenate all branches and add them to the body
            self.branches.add(self.b_0)
            self.branches.add(self.b_1)
            self.branches.add(self.b_2)
            self.body.add(self.branches)


class _InceptionResnetA(_InceptionResnetBlock):
    def __init__(
        self,
        name,
        in_ch,
        ch_0_0=32,
        ch_1_0=32,
        ch_1_1=32,
        ch_2_0=32,
        ch_2_1=48,
        ch_2_2=64,
        ch=384,
        bn_mom=0.9,
        act_type="relu",
        res_scale_fac=0.2,
        use_se=True,
        shortcut=True,
    ):
        """
        Definition of the InceptionResnetA block

        :param name: name prefix for all blocks
        :param ch_0_0: Number of channels for 1st conv operation in branch 0
        :param ch_1_0: Number of channels for 1st conv operation in branch 1
        :param ch_1_1: Number of channels for 2nd conv operation in branch 1
        :param ch_2_0: Number of channels for 1st conv operation in branch 2
        :param ch_2_1: Number of channels for 2nd conv operation in branch 2
        :param ch_2_2: Number of channels for 3rd conv operation in branch 2
        :param ch: Number of channels for conv operation after concatenating branches (no act is applied here)
        :param bn_mom: Batch normalization momentum parameter
        :param act_type: Activation type to use
        :param res_scale_fac: Constant multiply scalar which is applied to the residual activations maps
        :param shortcut: Decide weather to enable a shortcut connection
        """

        super(_InceptionResnetA, self).__init__(name, ch, res_scale_fac, act_type, bn_mom, use_se, shortcut)

        self.body = HybridSequential(prefix="")

        # entry point for all branches
        self.branches = HybridConcurrent(axis=1, prefix="")

        # branch 0 of block type A
        self.b_0 = HybridSequential()
        self.b_0.add(Conv2D(channels=ch_0_0, kernel_size=(1, 1), prefix="%s_0_conv0" % name, in_channels=in_ch))
        self.b_0.add(get_act(act_type, prefix="%s_0_%s0" % (name, act_type)))

        # branch 1 of block type A
        self.b_1 = HybridSequential()
        self.b_1.add(Conv2D(channels=ch_1_0, kernel_size=(1, 1), prefix="%s_1_conv0" % name, in_channels=in_ch))
        self.b_1.add(get_act(act_type, prefix="%s_1_%s0" % (name, act_type)))
        self.b_1.add(
            Conv2D(channels=ch_1_1, kernel_size=(3, 3), padding=(1, 1), prefix="%s_1_conv1" % name, in_channels=ch_1_0)
        )
        self.b_1.add(get_act(act_type, prefix="%s_1_%s1" % (name, act_type)))

        # branch 2 of block type A
        self.b_2 = HybridSequential()
        self.b_2.add(Conv2D(channels=ch_2_0, kernel_size=(1, 1), prefix="%s_2_conv0" % name, in_channels=in_ch))
        self.b_2.add(get_act(act_type, prefix="%s_2_%s0" % (name, act_type)))
        self.b_2.add(
            Conv2D(channels=ch_2_1, kernel_size=(3, 3), padding=(1, 1), prefix="%s_2_conv1" % name, in_channels=ch_2_0)
        )
        self.b_2.add(get_act(act_type, prefix="%s_2_%s1" % (name, act_type)))
        self.b_2.add(
            Conv2D(channels=ch_2_2, kernel_size=(3, 3), padding=(1, 1), prefix="%s_2_conv2" % name, in_channels=ch_2_1)
        )
        self.b_2.add(get_act(act_type, prefix="%s_2_%s2" % (name, act_type)))
        # self.b_2.add(PReLU(prefix='%s_2_%s2' % (name, act_type)))
        # concatenate all branches and add them to the body
        self.branches.add(self.b_0)
        self.branches.add(self.b_1)
        self.branches.add(self.b_2)
        self.body.add(self.branches)

        # apply a single CNN layer without activation function
        self.body.add(
            Conv2D(
                channels=ch,
                kernel_size=(1, 1),
                prefix="%s_conv0" % name,
                in_channels=ch_0_0 + ch_1_1 + ch_2_2,
                use_bias=False,
            )
        )


class _InceptionResnetB(_InceptionResnetBlock):
    def __init__(
        self,
        name,
        in_ch,
        ch_0_0=192,
        ch_1_0=128,
        ch_1_1=160,
        ch_1_2=192,
        ch=1152,
        bn_mom=0.9,
        act_type="relu",
        res_scale_fac=0.2,
        use_se=True,
        shortcut=True,
    ):
        """
        Definition of the InceptionResnetB block

        :param name: name prefix for all blocks
        :param ch_0_0: Number of channels for 1st conv operation in branch 0
        :param ch_1_0: Number of channels for 1st conv operation in branch 1
        :param ch_1_1: Number of channels for 2nd conv operation in branch 1
        :param ch_1_2: Number of channels for 3rd conv operation in branch 1
        :param ch: Number of channels for conv operation after concatenating branches (no act is applied here)
        :param bn_mom: Batch normalization momentum parameter
        :param act_type: Activation type to use
        :param res_scale_fac: Constant multiply scalar which is applied to the residual activations maps
        """
        super(_InceptionResnetB, self).__init__(name, ch, res_scale_fac, act_type, bn_mom, use_se, shortcut)

        self.body = HybridSequential(prefix="")

        # entry point for all branches
        self.branches = HybridConcurrent(axis=1, prefix="")

        # branch 0 of block type B
        self.b_0 = HybridSequential()
        self.b_0.add(Conv2D(channels=ch_0_0, kernel_size=(1, 1), prefix="%s_0_conv0" % name, in_channels=in_ch))
        self.b_0.add(get_act(act_type, prefix="%s_0_%s0" % (name, act_type)))

        # branch 2 of block type B
        self.b_1 = HybridSequential()
        self.b_1.add(Conv2D(channels=ch_1_0, kernel_size=(1, 1), prefix="%s_1_conv0" % name, in_channels=in_ch))
        self.b_1.add(get_act(act_type, prefix="%s_2_%s0" % (name, act_type)))
        # self.b_1.add(Conv2D(channels=ch_1_1, kernel_size=(1, 7), padding=(0, 3), prefix='%s_1_conv1' % name, in_channels=ch_1_0))
        self.b_1.add(
            Conv2D(channels=ch_1_1, kernel_size=(1, 5), padding=(0, 2), prefix="%s_1_conv1" % name, in_channels=ch_1_0)
        )
        self.b_1.add(get_act(act_type, prefix="%s_2_%s1" % (name, act_type)))
        # self.b_1.add(Conv2D(channels=ch_1_2, kernel_size=(7, 1), padding=(3, 0), prefix='%s_1_conv2' % name, in_channels=ch_1_1))
        self.b_1.add(
            Conv2D(channels=ch_1_2, kernel_size=(5, 1), padding=(2, 0), prefix="%s_1_conv2" % name, in_channels=ch_1_1)
        )
        self.b_1.add(get_act(act_type, prefix="%s_1_%s2" % (name, act_type)))

        # concatenate all branches and add them to the body
        self.branches.add(self.b_0)
        self.branches.add(self.b_1)
        self.body.add(self.branches)

        # apply a single CNN layer without activation function
        self.body.add(
            Conv2D(
                channels=ch, kernel_size=(1, 1), prefix="%s_conv0" % name, in_channels=ch_0_0 + ch_1_2, use_bias=False
            )
        )


class _InceptionResnetC(_InceptionResnetBlock):
    def __init__(
        self,
        name,
        in_ch,
        ch_0_0=192,
        ch_1_0=128,
        ch_1_1=224,
        ch_1_2=256,
        ch=2144,
        bn_mom=0.9,
        act_type="relu",
        res_scale_fac=0.2,
        use_se=True,
        shortcut=True,
    ):
        """
        Definition of the InceptionResnetC block

        :param name: name prefix for all blocks
        :param ch_0_0: Number of channels for 1st conv operation in branch 0
        :param ch_1_0: Number of channels for 1st conv operation in branch 1
        :param ch_1_1: Number of channels for 2nd conv operation in branch 1
        :param ch_1_2: Number of channels for 3rd conv operation in branch 1
        :param ch: Number of channels for conv operation after concatenating branches (no act is applied here)
        :param bn_mom: Batch normalization momentum parameter
        :param act_type: Activation type to use
        :param res_scale_fac: Constant multiply scalar which is applied to the residual activations maps
        """
        super(_InceptionResnetC, self).__init__(name, ch, res_scale_fac, act_type, bn_mom, use_se, shortcut)

        self.res_scale_fac = res_scale_fac
        self.block_name = name

        self.body = HybridSequential(prefix="")

        # entry point for all branches
        self.branches = HybridConcurrent(axis=1, prefix="")

        # branch 0 of block type C
        self.b_0 = HybridSequential()
        self.b_0.add(Conv2D(channels=ch_0_0, kernel_size=(1, 1), prefix="%s_0_conv0" % name, in_channels=in_ch))
        self.b_0.add(get_act(act_type, prefix="%s_0_%s0" % (name, act_type)))

        # branch 2 of block type C
        self.b_1 = HybridSequential()
        self.b_1.add(Conv2D(channels=ch_1_0, kernel_size=(1, 1), prefix="%s_1_conv0" % name, in_channels=in_ch))
        self.b_1.add(get_act(act_type, prefix="%s_2_%s0" % (name, act_type)))
        self.b_1.add(
            Conv2D(channels=ch_1_1, kernel_size=(1, 3), padding=(0, 1), prefix="%s_1_conv1" % name, in_channels=ch_1_0)
        )
        self.b_1.add(get_act(act_type, prefix="%s_2_%s1" % (name, act_type)))
        self.b_1.add(
            Conv2D(channels=ch_1_2, kernel_size=(3, 1), padding=(1, 0), prefix="%s_1_conv2" % name, in_channels=ch_1_1)
        )
        self.b_1.add(get_act(act_type, prefix="%s_1_%s2" % (name, act_type)))

        # concatenate all branches and add them to the body
        self.branches.add(self.b_0)
        self.branches.add(self.b_1)
        self.body.add(self.branches)

        # apply a single CNN layer without activation function
        self.body.add(
            Conv2D(
                channels=ch, kernel_size=(1, 1), prefix="%s_conv0" % name, in_channels=ch_0_0 + ch_1_2, use_bias=False
            )
        )
