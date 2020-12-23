#' Label a dataset using a supplied learner.
#'
#' Uses the provided learner in order to predict on the provided task's features.
#' `prob = TRUE` implements Dark Knowledge as implmented by Hinton, 2015.
#' If 'prob', predicts probabilities, else response.
#' The mixing parameter 'alpha' corresponds to a trade-off between the predicted target
#' and the actual target, where 0 only uses predicted targets (default) and 1 only uses the true label.
#'
#' @param task [`Task`] \cr
#' Task to create labels for.
#' @param teacher [`Learner`] \cr
#' Trained learner to compress into a neural network.
#' @param prob [`logical`] \cr
#' Predict probabilities instead of response? Default: True.
#' @param alpha [`numeric`] \cr
#' Specifies a convex-combination of labels predicted by the learner (alpha = 0)
#' and labels of the observation before swapping. Default: 0.
#'
#' @export
#' @return
#'   A [`matrix`] with nclasses cols and nobs rows containing probabilities.
label_artificial_data_classif = function(task, teacher, prob = TRUE, alpha = 0) {
  target = task$data(cols = task$target_names)[[task$target_names]]
  if (prob) {
    assert_true(teacher$predict_type == "prob")
    prd_tgt = teacher$predict(task)$prob
    if (alpha > 0) {
      true_tgt = keras::to_categorical(as.integer(target) - 1)
      prd_tgt = abs((1 - alpha) * true_tgt - alpha * prd_tgt)
    }
  } else {
    prd_tgt = teacher$predict(task)$response
    prd_tgt = keras::to_categorical(as.integer(prd_tgt) - 1)
  }
  return(prd_tgt)
}

#' Label a dataset using a supplied learner.
#'
#' Uses the provided learner in order to predict on the provided task's features.
#' `prob = TRUE` implements Dark Knowledge as implmented by Hinton, 2015.
#' If 'prob', predicts probabilities, else response.
#' The mixing parameter 'alpha' corresponds to a trade-off between the predicted target
#' and the actual target, where 0 only uses predicted targets (default) and 1 only uses the true label.
#'
#' @param task [`Task`] \cr
#' Task to create labels for.
#' @param teacher [`Learner`] \cr
#' Trained learner to compress into a neural network.
#' @param alpha [`numeric`] \¢r
#' Specifies a convex-combination of labels predicted by the learner (alpha = 0)
#' and labels of the observation before swapping.
#'
#' @export
#' @return
#'   A [`vector`] containing the response for each observation
label_artificial_data_regr = function(task, teacher, alpha = 0) {
  target = task$data(cols = task$target_names)[[task$target_names]]
  prd_tgt = teacher$predict(task)$response
  (1 - alpha) * target - alpha * prd_tgt
}