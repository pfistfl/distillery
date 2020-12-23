# For an input column swap the values in munge style.
munge_feature = function(feature, nns, switch_prob, var) {
  swap_ids = sample(seq_along(feature), size = ceiling(length(feature) * switch_prob), replace = FALSE)
  if (class(feature)[1] == "integer") {
    round(swap_numerics(swap_ids, feature, nns, var))
  } else if (class(feature)[1] == "numeric") {
    swap_numerics(swap_ids, feature, nns, var)
  } else {
    swap_factors(swap_ids, feature, nns, var)
  }
}


#' @title Munge a Task.
#'
#' @details
#' Always creates a new dataset exactly as large as 'task'.
#' @param task [`Task ]
#' Task to create artificial samples from.
#' @param switch_prob [`numeric`] \cr
#' Swapping probabilitiy. Default: 0.5.
#' @param var [`numeric`] \cr
#' For numeric features, a new value is sampled from N(x_nn, |x -x_nn| / var)
#' where x is a feature value, x_nn the nearest neighbour's corresponding feature value.
#' Default: 2.
#' @export
munge_task = function(task, switch_prob = 0.5, var = 2) {
  # Encode Data to compute NN's
  task_enc = (po("fixfactors") %>>% po("scale") %>>% po("encode"))$train(task)[[1]]
  data_enc = task_enc$data(cols = task_enc$feature_names)
  nns = FNN::get.knn(data_enc, k = 1)[[1]][,1]
  # Munge data
  y = task$data(cols=task$target_names)
  x = task$data(cols=task$feature_names)
  x_munged = mlr3misc::map_dtc(x, munge_feature, nns = nns, switch_prob = switch_prob, var = var)
  cbind(y, x_munged)
}


#' Generate artificial data, keeping 'keep_fraction' bootstrap samples.
#' Keeps the original label
#'
#' @param task [`Task ]
#' Task to create artificial samples from.
#' @param min_n [`integer`] \cr
#' Minimum number of observations to create. Default: 10000.
#' @param keep_fraction [`numeric`] \cr
#' Percentage of original data fraction to keep. Default: 0.1.
#' @param  ... `any` \cr
#' Passed on to `munge_task`.
#' @export
gen_artificial_data = function(task, min_n = 10000, keep_fraction = .1, ...) {

  drop = task$nrow  - (min_n * (1 - keep_fraction)) %% task$nrow
  # Bootstrap from original data
  boots = task$data(rows = sample(task$row_ids, ceiling(min_n * keep_fraction), replace = TRUE))

  # Munge: We always "munge" from the "munged" data
  tm = task$clone()
    while (tm$nrow < min_n) {
    x = munge_task(task = tm, ...)
    tm$rbind(x)
  }
  # Filter down to 'n' and append bootstrapped data
  tm$rbind(boots)
  return(tm)
}
