# Neural Networks

### A single neuron takes in multiple inputs (a vector) and produces one output (a scalar):
<p align="center">
  <img width="600" height="460" src="./img/fig1.png">
</p>

### Inside the neuron we take a linear combination of the inputs, add a bias, and apply a (nonlinear) activation function:
<p align="center">
  <img width="600" height="460" src="./img/fig2.png">
</p>

### The activation function behaves like a switch:
<p align="center">
  <img width="600" height="460" src="./img/fig3.png">
</p>

### And so a single neuron simply partitions the vector space. The weights control the direction of the partition:
<p align="center">
  <img width="600" height="460" src="./img/fig4.png">
</p>

### And the bias applies a shift:
<p align="center">
  <img width="600" height="460" src="./img/fig5.png">
</p>

### To create more partitions, we need more neurons:
<p align="center">
  <img width="600" height="460" src="./img/fig6.png">
</p>

### But now the output is a vector equal to number of neurons:
<p align="center">
  <img width="600" height="460" src="./img/fig7.png">
</p>

### We can represent the new space as node on a hypercube:
<p align="center">
  <img width="600" height="460" src="./img/fig8.png">
</p>

### An additional layer can be used to partition this new space.
<p align="center">
  <img width="600" height="460" src="./img/fig9.png">
</p>

### And the output of this second layer is again a hyperplane:
<p align="center">
  <img width="600" height="460" src="./img/fig10.png">
</p>

### The isolated datapoint [0,1] represents a full region in the first layer:
<p align="center">
  <img width="600" height="460" src="./img/fig11.png">
</p>

### Consider the following classification problem:
<p align="center">
  <img width="600" height="460" src="./img/fig12.png">
</p>

### We need three hyperplanes to separate the regions (3 neurons):
<p align="center">
  <img width="600" height="460" src="./img/fig13.png">
</p>

### The output is three dimensional:
<p align="center">
  <img width="600" height="460" src="./img/fig14.png">
</p>

### We can visualise these points on a three dimensional hypercube where we can color the nodes according to their class:
<p align="center">
  <img width="600" height="460" src="./img/fig15.png">
</p>

### A single neuron in a second layer can isolate the blue from the red:
<p align="center">
  <img width="600" height="460" src="./img/fig16.png">
</p>

### Which corresponds to the following partition:
<p align="center">
  <img width="600" height="460" src="./img/fig17.png">
</p>

### As we have shown, when using a step function, the first layer divides the space into regions. Two points in the same region are indistinguishable. And the second layer merges regions.

### As the classification problem gets more difficult, we need more neurons:
<p align="center">
  <img width="600" height="460" src="./img/fig18.png">
</p>

### And this is only a 2 dimensional problem. In practise, we use continous functions instead of the step function:
<p align="center">
  <img width="600" height="460" src="./img/fig19.png">
</p>

### The region is no longer mapped to two distinct points, and so subsequent layers do a lot more. Visualising the output of a single neuron is easy:
<p align="center">
  <img width="600" height="460" src="./img/fig20.png">
</p>

### Whereas visualising the output of two neurons is much more difficult. Consider another classification problem:
<p align="center">
  <img width="600" height="460" src="./img/fig21.png">
</p>

### We can classify this region with a 2-1 layer neural network with sigmoid activation functions. The first layer has two neurons
<p align="center">
  <img width="600" height="460" src="./img/fig22.png">
</p>
<p align="center">
  <img width="600" height="460" src="./img/fig23.png">
</p>

### And the second layer has one neuron, and the output is rounded.
<p align="center">
  <img width="600" height="460" src="./img/fig24.png">
</p>

### Which classifies the region like this:
<p align="center">
  <img width="600" height="460" src="./img/fig25.png">
</p>
