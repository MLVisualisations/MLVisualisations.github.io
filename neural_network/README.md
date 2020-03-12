# Neural Networks

### A single neuron takes in multiple inputs (a vector) and produces one output (a scalar):
<p align="center">
  <img width="300" height="300" src="./img/fig1.png">
</p>

### Inside the neuron we take a linear combination of the inputs, add a bias, and apply a (nonlinear) activation function:
<p align="center">
  <img width="300" height="300" src="./img/fig2.png">
</p>

### A biological neuron behaves like a switch, there is either an electical current of there is not. We can achieve the same using a step function:
<p align="center">
  <img width="600" height="300" src="./img/fig3.png">
</p>

### And so a single neuron simply partitions the vector space. The weights control the direction of the partition:
<p align="center">
  <img width="600" height="460" src="./img/fig4.png">
</p>

### And the bias applies a shift:
<p align="center">
  <img width="600" height="460" src="./img/fig5.png">
</p>

### A single neuron can only divide the space into two regions. To have more regions, we need more neurons. A layer of n neurons can create at most 2^n regions. Consider just 2 neurons:
<p align="center">
  <img width="400" height="400" src="./img/fig6.png">
</p>

### Each neuron produces a scalar output, and so the output of n neurons is a vector of length n. Each region is associated to a unique binary vector.
<p align="center">
  <img width="600" height="460" src="./img/fig7.png">
</p>

### The regions can be represented by nodes on a hypercube:
<p align="center">
  <img width="600" height="460" src="./img/fig8.png">
</p>

### We can apply additional layers to group the nodes, which in turn groups regions.
<p align="center">
  <img width="400" height="400" src="./img/fig9.png">
</p>

### The second layer also represents a hyperplane in the new space:
<p align="center">
  <img width="600" height="460" src="./img/fig10.png">
</p>

### The isolated node [0, 1] represents a full region in the first layer:
<p align="center">
  <img width="600" height="460" src="./img/fig11.png">
</p>

### Consider the following classification problem:
<p align="center">
  <img width="600" height="460" src="./img/fig12.png">
</p>

### We need three hyperplanes to carve up the space such that no region contains both a red and blue dot. This corresponds to a single layer with 3 neurons:
<p align="center">
  <img width="600" height="460" src="./img/fig13.png">
</p>

### The output is three dimensional binary vector:
<p align="center">
  <img width="600" height="460" src="./img/fig14.png">
</p>

### We can visualise these points on a three dimensional hypercube. The colour of the node represents the class of the notes in the region:
<p align="center">
  <img width="600" height="460" src="./img/fig15.png">
</p>

### A single hyperplane can isolate the blue nodes from the red nodes:
<p align="center">
  <img width="600" height="460" src="./img/fig16.png">
</p>

### Which corresponds to the following partition:
<p align="center">
  <img width="600" height="460" src="./img/fig17.png">
</p>

### As we have shown, when using a step function, the first layer divides the space into regions. Two points in the same region are indistinguishable. Subsequent layers simply merge regions. A more difficult classification problem would need more neurons:
<p align="center">
  <img width="600" height="460" src="./img/fig18.png">
</p>

### So far we have only considered a 2 dimensional vector space. As the input dimension increases, so would the number of neurons required in the first layer. In practise, we use continous functions instead of the step function. This allows backpropagation and makes subsequent layers more meaningful as we no longer have distinct regions after the first layer. Examples of commonly used activation functions include:
<p align="center">
  <img width="600" height="460" src="./img/fig19.png">
</p>

### Visualising the output of a single neuron with a continous activation function is easy:
<p align="center">
  <img width="600" height="460" src="./img/fig20.png">
</p>

### Whereas visualising the output of two neurons is much more difficult. Consider another classification problem:
<p align="center">
  <img width="600" height="460" src="./img/fig21.png">
</p>

### Suppose our first layer has two neurons with sigmoid activation function.
<p align="center">
  <img width="600" height="460" src="./img/fig22.png">
</p>

### We can visualise the space by mapping a color plane over it. Most of the points are shifted towards either [0, 0], [0, 1], [1, 0], [1, 1]:
<p align="center">
  <img width="600" height="300" src="./img/fig23.png">
</p>

### And a second layer with just one neuron again divides the space. If we round the output then we classify the region like so:
<p align="center">
  <img width="600" height="300" src="./img/fig24.png">
</p>

### A perfect divide. Trying to achieve such a classification would require more neurons if a step function was used. The sigmoid function can achieve anything a step function can by scaling the weights towards infinity:
<p align="center">
  <img width="600" height="300" src="./img/fig25.png">
</p>
