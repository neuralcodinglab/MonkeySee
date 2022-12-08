from typing import Any

from mxnet import autograd
from mxnet.gluon import Trainer
from mxnet.gluon.loss import L1Loss, Loss, SigmoidBCELoss
from mxnet.gluon.nn import Activation, BatchNorm, Conv2D, Conv2DTranspose, Dropout, HybridBlock, HybridSequential, \
    LeakyReLU
from mxnet.ndarray import NDArray, concat, full, mean, random_normal, zeros
from mxnet.gluon.model_zoo import vision
import mxnet as mx
import numpy as np
import mxnet.gluon.data.vision as datavision

class Lossfun:

    def __init__(self, alpha: float, beta_vgg:float, beta_pix: float, context=None) -> None:
        self._alpha = alpha
        self._bce = SigmoidBCELoss()
        self._beta_vgg = beta_vgg
        self._beta_pix = beta_pix
        self._l1 = L1Loss()
        self._vgg = VggLoss_Summed(context)


    def __call__(self, p: float, p_hat: NDArray, y: NDArray, y_hat: NDArray) -> NDArray:
        
        dis_loss = self._alpha * mean(self._bce(p_hat, full(p_hat.shape, p))) 
        
        gen_loss_vgg = self._beta_vgg * mean(self._vgg(y_hat, y))
        
        gen_loss_pix = self._beta_pix * mean(self._l1(y_hat, y))
        
        total_loss = dis_loss + gen_loss_vgg + gen_loss_pix
        
        return total_loss, dis_loss, gen_loss_vgg, gen_loss_pix
                                             
    
    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def bce(self) -> Loss:
        return self._bce

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def l1(self) -> Loss:
        return self._l1
    
    
class VggLoss_Summed():
    
    def __init__(self, context) -> None:
        
        self.vgg19=vision.vgg19(pretrained=True, ctx = context)
        self._l1 = L1Loss()
        self.transformer = datavision.transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        
    def __call__(self, y_hat, y):
        
        target_224 = mx.nd.contrib.BilinearResize2D(y, height=224, width=224)
        g_out_224 = mx.nd.contrib.BilinearResize2D(y_hat, height=224, width=224)

        target_224 = self.transformer(target_224)        
        g_out_224 = self.transformer(g_out_224)

        feat_target_c4 = self.vgg19.features[:8](target_224)
        feat_target_c7 = self.vgg19.features[:15](target_224)
        feat_target_c10 = self.vgg19.features[:22](target_224)
        feat_target_c13 = self.vgg19.features[:29](target_224)
        feat_target_c16 = self.vgg19.features[:35](target_224)

    

        feat_out_c4 = self.vgg19.features[:8](g_out_224)
        feat_out_c7 = self.vgg19.features[:15](g_out_224)
        feat_out_c10 = self.vgg19.features[:22](g_out_224)
        feat_out_c13 = self.vgg19.features[:29](g_out_224)
        feat_out_c16 = self.vgg19.features[:35](g_out_224)
    
        # taking L1 difference of each layers
        L_1 = self._l1(feat_out_c4, feat_target_c4) 
        L_2 = self._l1(feat_out_c7, feat_target_c7) 
        L_3 = self._l1(feat_out_c10, feat_target_c10) 
        L_4 = self._l1(feat_out_c13, feat_target_c13) 
        L_5 = self._l1(feat_out_c16, feat_target_c16) 

        summed_difference = (L_1 + L_2  + L_3 + L_4 + L_5)/5
        
        return summed_difference
    
    
class Layer(HybridBlock):
    def __init__(self) -> None:
        super(Layer, self).__init__()

    @property
    def count(self) -> int:
        raise NotImplementedError

    @property
    def depth(self) -> int:
        raise NotImplementedError

    def hybrid_forward(self, f: Any, x: NDArray, **kwargs) -> NDArray:
        raise NotImplementedError


class Identity(Layer):
    def __init__(self, count: int, depth: int) -> None:
        super(Identity, self).__init__()

        self._count = count
        self._depth = depth

    @property
    def count(self) -> int:
        return self._count

    @property
    def depth(self) -> int:
        return self._depth

    def hybrid_forward(self, f: Any, x: NDArray, **kwargs) -> NDArray:
        return x


class Skip(Layer):
    def __init__(self, count: int, depth: int, layer: Layer) -> None:
        super(Skip, self).__init__()

        with self.name_scope():
            self._block = HybridSequential()

            self._block.add(Conv2D(layer.depth, 4, 2, 1, use_bias=False, in_channels=depth))
            self._block.add(BatchNorm(momentum=0.1, in_channels=layer.depth))
            self._block.add(LeakyReLU(0.2))
            self._block.add(layer)
            self._block.add(Conv2DTranspose(count, 4, 2, 1, use_bias=False, in_channels=layer.count))
            self._block.add(BatchNorm(momentum=0.1, in_channels=count))

        self._count = count
        self._depth = depth
        self._layer = layer

    @property
    def block(self) -> HybridSequential:
        return self._block

    @property
    def count(self) -> int:
        return self._count + self._depth

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def layer(self) -> Layer:
        return self._layer

    def hybrid_forward(self, f: Any, x: NDArray, **kwargs) -> NDArray:
        return f.relu(f.concat(x, self._block(x), dim=1))


class Network(HybridSequential):
    def __init__(self, count: int, depth: int) -> None:
        super(Network, self).__init__()

        self._count = count
        self._depth = depth

        with self.name_scope():
            self.add(Conv2D(64, 4, 2, 1, in_channels=depth))
            self.add(LeakyReLU(alpha=0.2))

            layer = Identity(512, 512)
            layer = Skip(512, 512, layer)

            for _ in range(0):
                layer = Skip(512, 512, layer)

                layer.block.add(Dropout(0.5))

            layer = Skip(256, 256, layer)
            layer = Skip(128, 128, layer)
            layer = Skip(64, 64, layer)

            self.add(layer)
            self.add(Conv2DTranspose(count, 4, 2, 1, in_channels=128))
            self.add(Activation("sigmoid"))

        for param in self.collect_params().values():
            param.initialize()
            if "bias" in param.name:
                param.set_data(zeros(param.data().shape))
            elif "gamma" in param.name:
                param.set_data(random_normal(1, 0.02, param.data().shape))
            elif "weight" in param.name:
                param.set_data(random_normal(0, 0.02, param.data().shape))

    @property
    def count(self) -> int:
        return self._count

    @property
    def depth(self) -> int:
        return self._depth

class Generator_prediction:
    def __init__(self, input_channels, context) -> None:
        self._network = Network(3, input_channels)

    @property
    def network(self) -> HybridSequential:
        return self._network

    
class Generator:
    def __init__(self, input_channels, alpha: float, beta_vgg:float, beta_pix: float, context) -> None:
        self._lossfun = Lossfun(alpha, beta_vgg, beta_pix, context=context)
        self._network = Network(3, input_channels)
        self._trainer = Trainer(self._network.collect_params(), "adam", {
            "beta1": 0.5,
            "learning_rate": 0.0002
        })

    @property
    def lossfun(self) -> Lossfun:
        return self._lossfun

    @property
    def network(self) -> HybridSequential:

        return self._network

    @property
    def trainer(self) -> Trainer:
        return self._trainer

    def train(self, d: HybridSequential, x: NDArray, y: NDArray) -> float:

        with autograd.record():
            total_loss, dis_loss, gen_loss_vgg, gen_loss_pix = (lambda y_hat: self.lossfun(1, d(concat(x, y_hat, dim=1)), y, y_hat))(self._network(x))

        total_loss.backward()
        self.trainer.step(1)

        return float(total_loss.asscalar()), float(dis_loss.asscalar()), float(gen_loss_vgg.asscalar()), float(gen_loss_pix.asscalar())
