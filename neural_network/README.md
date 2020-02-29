# Neural Networks

### A single neuron takes in multiple inputs (a vector) and produces one output (a scalar):
<p align="center">
  <img width="460" height="460" src="./img/fig1.png">
</p>
### Inside the neuron we take a linear combination of the inputs, add a bias, and apply a (nonlinear) activation function:
![](./img/fig2.png)
### The activation function behaves like a switch:
![](./img/fig3.png)
### And so a single neuron simply partitions the vector space. The weights control the direction of the partition:
![](./img/fig4.png)
### And the bias applies a shift:
![](./img/fig5.png)
### To create more partitions, we need more neurons:
![](./img/fig6.png)
### But now the output is a vector equal to number of neurons:
![](./img/fig7.png)
### We can represent the new space as node on a hypercube:
![](./img/fig8.png)
### An additional layer can be used to partition this new space.
![](./img/fig9.png)
### And the output of this second layer is again a hyperplane:
![](./img/fig10.png)
### The isolated datapoint [0,1] represents a full region in the first layer:
![](./img/fig11.png)

### Consider the following classification problem:
![](./img/fig12.png)
### We need three hyperplanes to separate the regions (3 neurons):
![](./img/fig13.png)
### The output is three dimensional:
![](./img/fig14.png)
### We can visualise these points on a three dimensional hypercube where we can color the nodes according to their class:
![](./img/fig15.png)
### A single neuron in a second layer can isolate the blue from the red:
![](./img/fig16.png)
### Which corresponds to the following partition:
![](./img/fig17.png)
### As we have shown, when using a step function, the first layer divides the space into regions. Two points in the same region are indistinguishable. And the second layer merges regions.

### As the classification problem gets more difficult, we need more neurons:
![](./img/fig18.png)
### And this is only a 2 dimensional problem. In practise, we use continous functions instead of the step function:
![](./img/fig19.png)
### The region is no longer mapped to two distinct points, and so subsequent layers do a lot more.
