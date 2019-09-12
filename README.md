### Neural Network 

In the `recognize_digit` file, I used DNN method which is based on PaddlePaddle. 
This is a three layers multiple layer perceptron; two hidden layers which the sizes are 100, and the size of the output layer is 10, since the labels we have on hand is from 0-9. The activation function is Softmax, thus the output layer is also considered as a classifier. Therefore the structure of the network is: input layer ->> hidden layer ->> hidden layer ->> output layer.   
[The report could be found here](https://kyso.io/FFFlora/mnist-project/file/recognize_digit.ipynb)


--- 



### Logistic Regression vs XGBoost

In this project, I first use Logistic Regression to have a taste of how the classification goes, by that I have an accuracy of 83%; and then I try to use XGBoost, with the `xglinear` booster I have 85% accuracy, that is ~2% increase. Then I try the tree booster in XGBoost, finally reach 91.8% accuracy.

By applying `MLP` (Multi-layer Perceptron) classifier in `sklearn`, I use one hidden layer with 50 hidden units, run for 10 iterators at maximum, which gives an awesome result: 

```
Training set score: 0.986800
Test set score: 0.970000
```



**Neural Network wins :)** 

---



In this particular case, the Logistic Regression model prefers `l1` over `l2` penalty, since this mnist dataset has a very high sparsity.

Consider the vector ![](http://latex.codecogs.com/gif.latex?%5Cvec%7Bx%7D%20%3D%20%281%2C%5Cepsilon%29%20%5Cin%20%5CR%5E2) where ![](http://latex.codecogs.com/gif.latex?%5Cepsilon%20%3E%200) is small.  The `l1` and `l2` norms of  ![](http://latex.codecogs.com/gif.latex?%5Cvec%7Bx%7D), respectively, are given 

![](http://latex.codecogs.com/gif.latex?%5CVert%20%5Cvec%7Bx%7D%20%5CVert%20_1%20%3D%201&plus;%20%5Cepsilon%2C%20%5CVert%20%5Cvec%7Bx%7D%20%5CVert%20_2%5E2%20%3D%201&plus;%20%5Cepsilon%5E2)

Now say that, as part of some regularization procedure, we are going to reduce the magnitude of one of the elements of  vector x by  `δ ≤ ε`. If we change x1 to `1 - δ`, the  resulting norms are: 

![](http://latex.codecogs.com/gif.latex?%5CVert%20%5Cvec%7Bx%7D%20-%28%5Cdelta%2C0%29%5CVert_1%20%3D%201-%5Cdelta&plus;%20%5Cepsilon%2C%20%5CVert%20%5Cvec%7Bx%7D-%28%5Cdelta%2C0%29%20%5CVert%20_2%5E2%20%3D%201%20-2%5Cdelta%20&plus;%5Cdelta%5E2&plus;%20%5Cepsilon%5E2)

meanwhile, reduce x2 by  δ gives norms 

![](http://latex.codecogs.com/gif.latex?%5CVert%20%5Cvec%7Bx%7D%20-%280%2C%5Cdelta%29%5CVert_1%20%3D%201-%5Cdelta&plus;%20%5Cepsilon%2C%20%5CVert%20%5Cvec%7Bx%7D-%280%2C%5Cdelta%29%20%5CVert%20_2%5E2%20%3D%201%20-2%5Cepsilon%5Cdelta%20&plus;%5Cdelta%5E2&plus;%20%5Cepsilon%5E2)

Given the definitions that `l1`  loss is ![](http://latex.codecogs.com/gif.latex?l%28r%29%3D%7Cr%7C%3D%7Cy%20-%20%5Chat%7By%7D%7C), which gives the median regression; and`l2` loss or square loss is ![](http://latex.codecogs.com/gif.latex?l%28r%29%3Dr%5E2).

Normally large outliers have large residuals, and square loss gets much more effected by outliers than `l1` loss, that is, the penalty is huge if the error is large. 

Because of the special feature of this dataset, which is high in sparsity,  the 0's  in the sparse matrix will pull the loss function close to the x-axis; on the other hand, the reduction in `l1` norm is always equal to *δ*, regardless of the quantity being penalized.  Therefore choose `l1` penalty over `l2`. 





---

### Dataset

[mnist_784 ](https://www.openml.org/d/554)

>It is a subset of a larger set available from NIST. The digits have  been size-normalized and centered in a fixed-size image. It is a good  database for people who want to try learning techniques and pattern  recognition methods on real-world data while spending minimal efforts on  preprocessing and formatting. The original black and white (bilevel)  images from NIST were size normalized to fit in a 20x20 pixel box while  preserving their aspect ratio. The resulting images contain grey levels  as a result of the anti-aliasing technique used by the normalization  algorithm. the images were centered in a 28x28 image by computing the  center of mass of the pixels, and translating the image so as to  position this point at the center of the 28x28 field.
>
>With some classification methods (particularly template-based  methods, such as SVM and K-nearest neighbors), the error rate improves  when the digits are centered by bounding box rather than center of mass.  If you do this kind of pre-processing, you should report it in your  publications. The MNIST database was constructed from NIST's NIST  originally designated SD-3 as their training set and SD-1 as their test  set. However, SD-3 is much cleaner and easier to recognize than SD-1.  The reason for this can be found on the fact that SD-3 was collected  among Census Bureau employees, while SD-1 was collected among  high-school students. Drawing sensible conclusions from learning  experiments requires that the result be independent of the choice of  training set and test among the complete set of samples. Therefore it  was necessary to build a new database by mixing NIST's datasets.


