
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
t$set_row_role(sample(t$row_ids, 50), "validation")
l = lrn("classif.rpart", predict_type = "prob")$train(t)
```

we can evaluate the current learner on our `Task`.
```r
l$predict(t, t$row_roles$validation)$score()
```

Now we just call `distill` on our trained learner and `distillery` automatically trains and tunes the Student Network.

```r
library(distillery)
library(mlr3keras)
c = distill(l, t, budget = 1L)
```

Et voila, we can get an equivalent model, this time compressed into a neural network.

```r
c$predict(t, t$row_roles$validation)$score()
```

and look at the training trace:

```r
c$learner$plot()
```

## Compressing a Pipeline

As an example, we compress the pipeline from this [mlr3gallery blog post](https://mlr3gallery.mlr-org.com/posts/2020-04-27-tuning-stacking/).
```r
library(mlr3)
sonar_task = tsk("sonar")
sonar_task$set_row_role(sample(sonar_task$row_ids, 50), "validation")
sonar_task$col_roles$stratum = sonar_task$target_names
```

```r
library(mlr3filters)
library(mlr3pipelines)
# Level 0
rprt_lrn  = lrn("classif.rpart", predict_type = "prob")
glmnet_lrn =  lrn("classif.glmnet", predict_type = "prob")
lda_lrn = lrn("classif.lda", predict_type = "prob")
rprt_cv1 = po("learner_cv", rprt_lrn, id = "rprt_1")
glmnet_cv1 = po("learner_cv", glmnet_lrn, id = "glmnet_1")
lda_cv1 = po("learner_cv", lda_lrn, id = "lda_1")
anova = po("filter", flt("anova"), id = "filt1", filter.frac = 08)
mrmr = po("filter", flt("mrmr"), id = "filt2", filter.frac = 0.9)
find_cor = po("filter", flt("find_correlation"), id = "filt3", filter.frac = 0.9)
level0 = gunion(list(
  anova %>>% rprt_cv1,
  mrmr %>>% glmnet_cv1,
  find_cor %>>% po("removeconstants") %>>% lda_cv1,
  po("nop", id = "nop1")))  %>>%
  po("featureunion", id = "union1")

# Level 1
rprt_cv2 = po("learner_cv", rprt_lrn , id = "rprt_2")
glmnet_cv2 = po("learner_cv", glmnet_lrn, id = "glmnet_2")
lda_cv2 = po("learner_cv", lda_lrn, id = "lda_2")
level1 = level0 %>>%
  po("copy", 4) %>>%
  gunion(list(
    po("pca", id = "pca2_1", param_vals = list(scale. = TRUE)) %>>% rprt_cv2,
    po("pca", id = "pca2_2", param_vals = list(scale. = TRUE)) %>>% glmnet_cv2,
    po("pca", id = "pca2_3", param_vals = list(scale. = TRUE)) %>>% po("removeconstants", id = "rmcst") %>>% lda_cv2,
    po("nop", id = "nop2"))
  )  %>>%
  po("featureunion", id = "union2")
# Level 2
ranger_lrn = lrn("classif.ranger", predict_type = "prob")
ensemble = level1 %>>% ranger_lrn
ens_lrn = GraphLearner$new(ensemble)
```

We can now train and score:
```r
ens_lrn$train(sonar_task)
ens_lrn$predict(sonar_task, sonar_task$row_roles$validation)$score()
```


```r
c = distill(ens_lrn, sonar_task, budget = 1L)
c$predict(sonar_task, sonar_task$row_roles$validation)$score()
```
