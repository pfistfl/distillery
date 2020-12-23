#' Create train / validation data generators from a task and params
#'
#' @param task [`Task`]\cr
#'   An mlr3 [`Task`].
#' @param x_transform [`function`]\cr
#'   Function used to transform data to a keras input format for features.
#' @param y_transform [`function`]\cr
#'   Function used to transform data to a keras input format for the response.
#' @param validation_split [`numeric`]\cr
#'   Fraction of data to use for validation.
#' @param batch_size [`integer`]\cr
#'   Batch_size for the generators.
#' @param ... `any`\cr
#'   Additional arguments passed on to `gen_artificial_data`
#' @export
make_munge_generators = function(task, x_transform, y_transform, validation_split = 1/3, batch_size = 128L, seed = NULL, ...) { # nocov start
  args = list(...)

  # Default to a seed in case none is given
  if (is.null(seed)) seed = 123L

  task = gen_artificial_data(task, min_n = args$min_n , keep_fraction = args$keep_fraction, switch_prob = args$switch_prob, var = args$var)
  x = as.matrix(task$data(cols=task$feature_names))
  y = label_artificial_data_classif(task, teacher = args$teacher, prob = args$prob, alpha = args$alpha)
  gen = keras::image_data_generator(validation_split = validation_split)

  train_gen = mlr3keras::make_generator_from_xy(x, y, generator=gen, batch_size=as.integer(batch_size),
    shuffle=TRUE, seed=as.integer(seed), subset="training", ignore_class_split=TRUE)
  train_steps = train_gen$`__len__`()

  if (validation_split > 0) {
    valid_gen = mlr3keras::make_generator_from_xy(x, y, generator=gen, batch_size=as.integer(batch_size),
      shuffle=TRUE, seed=as.integer(seed), subset="validation", ignore_class_split=TRUE)
    valid_steps = valid_gen$`__len__`()
  } else {
    valid_gen = NULL
    valid_steps = NULL
  }
  list(train_gen = train_gen, valid_gen = valid_gen, train_steps = train_steps, valid_steps = valid_steps)
} # nocov end
