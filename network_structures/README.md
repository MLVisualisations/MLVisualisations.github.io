# Neural Structures

### Many different neural network architectures exist. Unsuprisingly, most of them are just standard deep neural networks with hardwired weights or activation functions.

## Convolution
### A convolution layer has proven to help neural networks when dealing with images. The idea is that a single pixel should only be affected by neighbouring pixels and not pixels far away. This would involve setting many weights to zero. This can be taken one step further by enforcing many of the existing weights share values. An example of the convolution operator is shown here:
<p align="center">
  <img width="600" height="300" src="./img/convolution1.png">
</p>
### Which can be represented by a standard neural network layer. The weights are represented by the arrows. Two arrows of the same colour represent the same weight and no arrow implies a zero weight.
<p align="center">
  <img width="600" height="300" src="./img/convolution1.png">
</p>
