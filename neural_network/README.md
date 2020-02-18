# Neural Networks

### A neuron takes multiple inputs and produces one output:
![](./img/fig1.png)
### Inside the neuron we take a weighted sum, add a bias, and apply an activation function:
![](./img/fig2.png)
### The activation function behaves like a switch:
![](./img/fig3.png)
### And so a neuron partitions the input space where the weights control the direction:
![](./img/fig4.png)
### And the bias applies a shift:
![](./img/fig5.png)
### To create more partitions, we need more neurons:
![](./img/fig6.png)
### And the output is a vector equal to number of neurons:
![](./img/fig7.png)
### We can represent this space as node on a hypercube:
![](./img/fig8.png)
### We could add an additional layer to partition this space.
### Alternatively, we could sum the output of multiple neurons:
![](./img/fig9.png)
### This combines regions:
![](./img/fig10.png)
### And we can keep adding neurons:
![](./img/fig11.png)
### Suppose we wanted to classify the green crosses, with 4 neurons we could have:
![](./img/fig12.png)
### And with 8 neurons:
![](./img/fig13.png)
### If the classification problem became more complex, we could need many neurons:
![](./img/fig14.png)

### In practise, we use continous functions which approximate the step function
![](./img/fig15.png)
![](./img/fig16.png)
### And now the regions become less distinct
![](./img/fig17.png)
![](./img/fig18.png)
