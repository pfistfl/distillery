context("Distillation works for classification")

test_that("Classification", {
  skip_if_not_installed("mlr3learners")
  skip_if_not_installed("glmnet")
  t = tsk("iris")
  l = lrn("classif.glmnet", predict_type = "prob")$train(t)
  c = LearnerClassifKerasDistill$new(l)
  c$param_set$values$callbacks = mlr3keras::cb_es(patience = 50L)
  c$param_set$values$epochs = 200
  c$param_set$values$n = 500
  c$param_set$values$layer_units = NULL
  c$param_set$values$use_embedding = FALSE
  c$param_set$values$keep_fraction = 1
  c$param_set$values$optimizer = keras::optimizer_sgd(lr=0.1, momentum=0.9)
  c$train(t)
  expect_true(mean((c$predict(t)$prob - l$predict(t)$prob)^2) < 0.05)
  expect_true(mean(c$predict(t)$response == l$predict(t)$response) > 0.9)
})

context("Distillation works for regression")

test_that("swap_factors", {
  t = tsk("mtcars")
  l = lrn("regr.glmnet")$train(t)
  c = LearnerRegrKerasDistill$new(l)
  c$param_set$values$callbacks = mlr3keras::cb_es(patience = 50L)
  c$param_set$values$epochs = 200
  c$param_set$values$n = 500
  c$param_set$values$layer_units = NULL
  c$param_set$values$use_embedding = FALSE
  c$param_set$values$keep_fraction = 1
  c$train(t)
  c$predict(t)
})