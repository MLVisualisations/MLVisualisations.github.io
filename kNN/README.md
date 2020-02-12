# k Nearest Neighbours

### Consider two classes, blue and red:
![](./img/fig1.png)
### Well which class does the green cross belong to?
![](./img/fig2.png)
### We could look at its nearest neighbour:
![](./img/fig3.png)
### Using this strategy divides the space into two regions:
![](./img/fig4.png)
### Outliers can damage the regions:
![](./img/fig5.png)
### We can remove them by considering more neighbour:
![](./img/fig6.png)
### But considering too many neighbours is also bad:
![](./img/fig7.png)
### The complexity of the algorithm increases as we add more datapoints:
![](./img/fig8.png)
### Which class does the green cross belong to?
![](./img/fig9.png)
### Well we first must compute all these distances:
![](./img/fig10.png)
### And large regions of the space may have no nearby neighbours!
![](./img/fig11.png)
