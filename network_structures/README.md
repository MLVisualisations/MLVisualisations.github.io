# Neural Structures

### Many different neural network architectures exist. Unsuprisingly, most of them are just standard deep neural networks with hardwired weights or activation functions.

## Convolution
### A convolution layer has proven to help neural networks when dealing with images. The idea is that a single pixel should only be affected by neighbouring pixels and not pixels far away. This would involve setting many weights to zero. This can be taken one step further by enforcing many of the existing weights share values. An example of the convolution operator is shown here:
<p align="center">
  <img width="800" height="300" src="./img/convolution1.png">
</p>
Image credit: https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/blocks/2d-convolution-block

### Which can be represented by a standard neural network layer. The weights are represented by the arrows. Two arrows of the same colour represent the same weight and no arrow implies a zero weight.
<p align="center">
  <img width="600" height="300" src="./img/convolution2.png">
</p>

## Pooling
### Pooling is often used along side convolution layers. The idea is to take a small block of the images and squish it into one value.
<p align="center">
  <img width="500" height="300" src="./img/pooling1.png">
</p>

### If we do an average pooling, then the layer can be represented by a single layer. If we wanted to do max pooling then we would have to introduce layer operations. Remember activation functions inside neurons are applied to a scalar, not a vector.
<p align="center">
  <img width="700" height="300" src="./img/pooling2.png">
</p>

## Residual blocks
### ResNet consists of residual blocks. Residual block prevent the vanishing gradient problem by allowing gradients to leap frog layer. The most common type of residual block is this:
<p align="center">
  <img width="500" height="300" src="./img/ResNet1.png">
</p>

### Again this can be represented by standard neural network if we introduce neurons with the identifiy as an activation function (red) and edges with weight equal to one (red):
<p align="center">
  <img width="500" height="300" src="./img/ResNet2.png">
</p>
