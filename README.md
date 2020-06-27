# TF.Keras-常用型号

__自己整理的一些tensorflow下ķeras实现的模型,可在Tensorflow2.X下运行__

## 提示：以下模型均不包含预训练权重的载入，只是模型的实现；不同的卷积模块大部分在分类分割模型中已包含。

## 分类模型：
* AlexNet
* Darknet53
* DenseNet
* Dual_path_network
* GoogleNet
* MNasNet
* Resnet34
* Resnet50
* SEResNeXt
* VGG16
* Squeeze_Excite-Network
* MobileNetV3
* Efficientnet
* SE_HRNet
* ResNest

## 分割模型：
* FCN8S
* ICNet
* MiniNetv2
* PSPNet-ResNet50
* RAUNet-3D
* Refinenet
* Segnet
* Unet
* Unet_Xception_Resnetblock
* ResNextFPN
* Deeplabv2
* Deeplabv3+
* FastFCN
* HRNet
* ResUNet-a
* RCNN-UNet
* Attention Unet
* RCNN-Attention Unet
* UNet ++
### Unet_family:
#### 不同种类的Unet模型图像分割的实现
1、UNet -U-Net：用于生物医学图像分割的卷积网络 https://arxiv.org/abs/1505.04597 <br>
2、RCNN-UNet-基于U-Net的递归残积卷积神经网络（R2U-Net）用于医学图像分割 https://arxiv.org/abs/1802.06955 <br>
3、Attention Unet -Attention U-Net：学习在哪里寻找胰腺 https://arxiv.org/abs/1804.03999 <br>
4、RCNN-Attention Unet -Attention R2U-Net：只需将两个最新的高级作品集成在一起（R2U-Net + Attention U-Net） <br>
5、嵌套的UNet -UNet ++：用于医学图像分割的嵌套U-Net体系结构 https://arxiv.org/abs/1807.10165 <br>
#### 参考:
[Unet-Segmentation-Pytorch-Nest-of-Unets](https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets) <br>
不同点:我的实现初始滤波数32，原始为64。

### 分割损失函数：
* Focal_Tversky_loss
* C_Focal_loss
* B_Focal_loss
* LovaszSoftmax
* WeightedCCE
* jaccard_loss
* bce_jaccard_loss
* cce_jaccard_loss
* dice_loss
* bce_dice_loss
* cce_dice_loss

### 分割指标：
* iou_score
* jaccard_score
* f1_score
* f2_score
* dice_score

### 新型激活函数：
* gelu
* swish
* mish

### 卷积模块：
* SE
* Res2Net
* Deformable_Conv

### Layer：
* FRN
* attention（PAM空间注意力和CAM通道注意力）
* BiFPN

### Others：
* TCN（时间卷积网络——解决LSTM的并发问题）
