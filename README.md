### Logistic Regression vs XGBoost

In this project, I first use Logistic Regression to have a taste of how the classification goes, by that I have an accuracy of 83%; and then I try to use XGBoost, with the `xglinear` booster I have 85% accuracy, that is ~2% increase. Then I try the tree booster in XGBoost, finally reach 91.8% accuracy.

---



In this particular case, the Logistic Regression model prefers `l1` over `l2` penalty, since this mnist dataset has a very high sparsity.

Consider the vector ![](http://latex.codecogs.com/gif.latex?%5Cvec%7Bx%7D%20%3D%20%281%2C%5Cepsilon%29%20%5Cin%20%5CR%5E2) where ![](http://latex.codecogs.com/gif.latex?%5Cepsilon%20%3E%200) is small.  The `l1` and `l2` norms of  ![](http://latex.codecogs.com/gif.latex?%5Cvec%7Bx%7D), respectively, are given 

![](http://latex.codecogs.com/gif.latex?%5CVert%20%5Cvec%7Bx%7D%20%5CVert%20_1%20%3D%201&plus;%20%5Cepsilon%2C%20%5CVert%20%5Cvec%7Bx%7D%20%5CVert%20_2%5E2%20%3D%201&plus;%20%5Cepsilon%5E2)

Now say that, as part of some regularization procedure, we are going to reduce the magnitude of one of the elements of $\vec{x} $ by $\delta\le \epsilon$. If we change $x_1$ to $1-\delta$, the  resulting norms are: 

![](http://latex.codecogs.com/gif.latex?%5CVert%20%5Cvec%7Bx%7D%20-%28%5Cdelta%2C0%29%5CVert_1%20%3D%201-%5Cdelta&plus;%20%5Cepsilon%2C%20%5CVert%20%5Cvec%7Bx%7D-%28%5Cdelta%2C0%29%20%5CVert%20_2%5E2%20%3D%201%20-2%5Cdelta%20&plus;%5Cdelta%5E2&plus;%20%5Cepsilon%5E2)

meanwhile, reduce $x_2$ by $\delta$ gives norms

![](http://latex.codecogs.com/gif.latex?%5CVert%20%5Cvec%7Bx%7D%20-%280%2C%5Cdelta%29%5CVert_1%20%3D%201-%5Cdelta&plus;%20%5Cepsilon%2C%20%5CVert%20%5Cvec%7Bx%7D-%280%2C%5Cdelta%29%20%5CVert%20_2%5E2%20%3D%201%20-2%5Cepsilon%5Cdelta%20&plus;%5Cdelta%5E2&plus;%20%5Cepsilon%5E2)

Given the definitions that `l1`  loss is ![](http://latex.codecogs.com/gif.latex?l%28r%29%3D%7Cr%7C%3D%7Cy%20-%20%5Chat%7By%7D%7C), which gives the median regression; and`l2` loss or square loss is ![](http://latex.codecogs.com/gif.latex?l%28r%29%3Dr%5E2).

Normally large outliers have large residuals, and square loss gets much more effected by outliers than `l1` loss, that is, the penalty is huge if the error is large. 

Because of the special feature of this dataset, which is high in sparsity,  the 0's  in the sparse matrix will pull the loss function close to the x-axis; on the other hand, the reduction in `l1` norm is always equal to *δ*, regardless of the quantity being penalized.  Therefore choose `l1` penalty over `l2`. 