
# distillery

<!-- badges: start -->
<!-- badges: end -->

The goal of distillery is to compress arbitrary `mlr3` `Learners`, `Pipelines` and `Ensembles` into a
single, possibly leightweight Neural Network using `keras`.

## Installation

You can install the released version of distillery from [CRAN](https://CRAN.R-project.org) with:

``` r
remotes::install_github("pfistfl/distillery")
```

## Example

This is a basic example which shows you how to solve a common problem:


Assume we have a trained learner, which we want to compress into a Neural Network.
This can also be complete ML Pipelines and Ensembles!

```r
library(mlr3)
t = tsk("iris")
l = lrn("classif.rpart", predict_type = "prob")$train(t)
```
we can evaluate the current learner on our `Task`.
```r
l$predict(t)$score()
```

Now we just cerate a `LearnerClassifKerasDistill` and train it on the `Task`.

```r
library(distillery)
library(mlr3keras)
c = LearnerClassifKerasDistill$new(l)
c$train(t)
```

Et voila, we can get an equivalent model, this time compressed into a neural network.

```r
c$predict(t)$score()
```
