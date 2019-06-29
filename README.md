### Logistic Regression vs XGBoost

In this project, I first use Logistic Regression to have a taste of how the classification goes, by that I have an accuracy of 83%; and then I try to use XGBoost, with the `xglinear` booster I have 85% accuracy, that ~2% increase. Then I try the tree booster in XGBoost, finally have 91.8% accuracy.

---



In this particular case, with Logistic Regression model, It prefers `l1` to `l2` penalty, since this mnist dataset has a very high sparsity.

Consider the vector $\vec{x} = (1,\epsilon) \in \R^2 $ where *ε*>0 is small.  The $l1$ and $l2$ norms of $\vec{x}$ , respectively, are given 

$$\Vert \vec{x} \Vert _1 = 1+ \epsilon, \Vert \vec{x} \Vert _2^2 = 1+ \epsilon^2 $$



Now say that, as part of some regularization procedure, we are going to reduce the magnitude of one of the elements of $\vec{x} $ by $\delta\le \epsilon$. If we change $x_1$ to $1-\delta$, the  resulting norms are: 

$$ \Vert \vec{x} -(\delta,0)\Vert_1  = 1-\delta+ \epsilon, \Vert \vec{x}-(\delta,0) \Vert _2^2 = 1 -2\delta +\delta^2+ \epsilon^2 $$

meanwhile, reduce $x_2$ by $\delta$ gives norms

$$ \Vert \vec{x} -(0,\delta)\Vert_1  = 1-\delta+ \epsilon, \Vert \vec{x}-(0,\delta) \Vert _2^2 = 1 -2\epsilon\delta +\delta^2+ \epsilon^2 $$

Given the definitions that $l1$  loss is $l(r)=|r|=|y - \hat{y}|$ , which gives the median regression, and $l2$ loss or square loss is $l(r)=r^2$.

Normally large outliers have large residuals, and square loss gets much more effected by outliers than $l1$ loss, that is, the penalty is huge is error is large. 

Because of the special feature of this dataset, which is high in sparsity, thus the 0's  in the sparse matrix will pull the loss function to the x-axis, on the other hand, the reduction in $l1$ norm is always equal to *δ*, regardless of the quantity being penalized.  Therefore choose $l1$ penalty over $l2$. 