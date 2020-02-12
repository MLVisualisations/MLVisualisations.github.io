# Neural Networks

### A neuron takes multiple inputs and produces one output:
![](./img/fig1.png)
### Inside the neuron we take a weighted sum, add a bias, and apply an activation function:
![](./img/fig2.png)
### The activation function behaves like a switch:
![](./img/fig3.png)
### And so a neuron partitions the input space, the weights control the direction:
![](./img/fig4.png)
### And the bias applies a shift:
![](./img/fig5.png)

### Summing multiple neurons allows us to apply multiple partitions:
![](./img/fig6.png)
### And with enough neurons, we can partition any space:
![](./img/fig7.png)
![](./img/fig8.png)
### This result is known as the 'Universal Approximation Theorem'. The problem is, we could need many partitions.


### In practise, we use continous functions which approximate the step function
![](./img/fig8.png)
