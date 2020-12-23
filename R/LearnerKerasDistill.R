#' @title Keras Feed Forward Neural Network for Knowledge Distillation
#'
#' @description
#' Internally, samples new observations from the task data using a nonparametric sampling
#' technique (here: Munge (c.f. Caruana et al., 2011 Model Compression).
#' Those observations are then labeled using the supplied `teacher`; a fitted `mlr3` learner.
#' In addition to all hyperparameters from `kerasff`, this learner has the following
#' hyperparameters which allow for control over the distillation process:
#' * `n` :: number of artificial rows to sample
#' * `switch_prob` :: probability to switch out each entry with it's nearest neighbour
#' * `var` :: For numeric features, a new value is sampled from N(x_nn, |x -x_nn| / var)
#'   where x is a feature value, x_nn the nearest neighbour's corresponding feature value.
#' * `probabilities` :: Should probabilities be approximated instead of the response?
#'   (c.f. Hinton, 2015 Dark Knowledge)
#' * `alpha` :: Specifies a convex-combination of labels predicted by the learner (alpha = 0)
#'   and labels of the observation before swapping.
#'
#' @usage NULL
#' @aliases mlr_learners_classif.kerasdistill
#' @format [R6::R6Class()] inheriting from [mlr3keras::LearnerClassifKerasFF].
#'
#' @section Construction:
#' ```
#' LearnerClassifKerasDistill$new()
#' mlr3::mlr_learners$get("classif.kerasdistill")
#' mlr3::lrn("classif.kerasdistill")
#' ```
#' @template kerasff_description
#' @template learner_methods
#' @template seealso_learner
#' @templateVar learner_name classif.kerasdistill
#' @template example
#' @export
LearnerClassifKerasDistill = R6::R6Class("LearnerClassifKerasDistill",
  inherit = mlr3keras::LearnerClassifKerasFF,
  public = list(

    #' @field teacher [`Learner`] \cr
    #' Trained method to distill into a neural network
    teacher = NULL,

    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @param teacher [`Learner`] \cr
    #' Trained learner to compress into a neural network.
    initialize = function(teacher) {
      self$teacher = assert_learner(teacher)
      super$initialize()
      self$param_set$add(
        ParamSet$new(list(
          ParamInt$new("min_n", lower = 1, upper = Inf, tags = c("train", "distill")),
          ParamDbl$new("switch_prob", lower = 0, upper = 1, tags = c("train", "distill")),
          ParamDbl$new("var", lower = 0, upper = Inf, tags = c("train", "distill")),
          ParamLgl$new("probabilities", default = TRUE, tags = c("train", "distill")),
          ParamDbl$new("alpha", lower = 0, upper = 1, default = 0, tags = c("train", "distill")),
          ParamDbl$new("keep_fraction", lower = 0, upper = 1, default = 0.1, tags = c("train", "distill"))
      )))
      self$param_set$values = c(self$param_set$values, list(min_n = 1e4, switch_prob = 0.5, var = 2, probabilities = TRUE, alpha = 0, keep_fraction = 0.1))
      self$param_set$values$loss = "mean_squared_error"
      if (self$teacher$predict_type == "prob")
        self$predict_type = "prob"
      self$architecture$transforms$y = function(y, pars, model_loss) y
  }),
  private = list(
    #' @description
    #' Internal training function, runs the distillation
    #' @param task [`Task`] \cr
    #'   Data the `teacher` was trained on
    #' @return A list with slots 'model', 'history' and 'class_names'
    .train = function(task) {

      distill_pars = self$param_set$get_values(tags = "distill")
      pars = self$param_set$get_values(tags = "train")
      model = self$architecture$get_model(task, pars)

      # Either fit directly on data or create a generator and fit from there
      if (!pars$low_memory) {
        task = gen_artificial_data(task, min_n = distill_pars$min_n, keep_fraction = distill_pars$keep_fraction, switch_prob = distill_pars$switch_prob, var = distill_pars$var)
        features = task$data(cols = task$feature_names)
        x = self$architecture$transforms$x(features, pars)
        y = label_artificial_data_classif(task, teacher = self$teacher, prob = distill_pars$probabilities, alpha = distill_pars$alpha)
        y = self$architecture$transforms$y(y, pars = pars, model_loss = model$loss)
        history = invoke(keras::fit,
          object = model,
          x = x,
          y = y,
          epochs = as.integer(pars$epochs),
          class_weight = pars$class_weight,
          batch_size = as.integer(pars$batch_size),
          validation_split = pars$validation_split,
          verbose = as.integer(pars$verbose),
          callbacks = pars$callbacks
        )
      } else {
        generators = make_munge_generators(
          task = task,
          x_transform = function(features) self$architecture$transforms$x(features, pars = pars),
          y_transform = function(target) self$architecture$transforms$y(target, pars = pars, model_loss = model$loss),
          validation_split = pars$validation_split,
          batch_size = pars$batch_size,
          min_n = distill_pars$min_n,  keep_fraction = distill_pars$keep_fraction,
          switch_prob = distill_pars$switch_prob, var = distill_pars$var,
          teacher = self$teacher, prob = distill_pars$probabilities, alpha = distill_pars$alpha
        )
        history = invoke(keras::fit_generator,
          object = model,
          generator = generators$train_gen,
          epochs = as.integer(pars$epochs),
          class_weight = pars$class_weight,
          steps_per_epoch = generators$train_steps,
          validation_data = generators$valid_gen,
          validation_steps = generators$valid_steps,
          verbose = pars$verbose,
          callbacks = pars$callbacks)
      }
      return(list(model = model, history = history, class_names = task$class_names))
    }
  )
)


#' @title Keras Feed Forward Neural Network for Knowledge Distillation
#'
#' @description
#' Internally, samples new observations from the task data using a nonparametric sampling
#' technique (here: Munge (c.f. Caruana et al., 2011 Model Compression).
#' Those observations are then labeled using the supplied `teacher`; a fitted `mlr3` learner.
#' In addition to all hyperparameters from `kerasff`, this learner has the following
#' hyperparameters which allow for control over the distillation process:
#' * `n` :: number of artificial rows to sample
#' * `switch_prob` :: probability to switch out each entry with it's nearest neighbour
#' * `var` :: For numeric features, a new value is sampled from N(x_nn, |x -x_nn| / var)
#'   where x is a feature value, x_nn the nearest neighbour's corresponding feature value.
#' * `alpha` :: Specifies a convex-combination of labels predicted by the learner (alpha = 0)
#'   and labels of the observation before swapping.
#'
#' @usage NULL
#' @aliases mlr_learners_classif.kerasdistill
#' @format [R6::R6Class()] inheriting from [mlr3keras::LearnerClassifKerasFF].
#'
#' @section Construction:
#' ```
#' LearnerregrKerasDistill$new()
#' mlr3::mlr_learners$get("regr.kerasdistill")
#' mlr3::lrn("regr.kerasdistill")
#' ```
#' @template kerasff_description
#' @template learner_methods
#' @template seealso_learner
#' @templateVar learner_name regr.kerasdistill
#' @template example
#' @export
LearnerRegrKerasDistill = R6::R6Class("LearnerRegrKerasDistill",
  inherit = mlr3keras::LearnerRegrKerasFF,
  public = list(

    #' @field teacher [`Learner`] \cr
    #' Trained method to distill into a neural network
    teacher = NULL,

    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @param teacher [`Learner`] \cr
    #' Trained learner to compress into a neural network.
    initialize = function(teacher) {
      self$teacher = assert_learner(teacher)
      super$initialize()
      self$param_set$add(ParamSet$new(list(
        ParamInt$new("min_n", lower = 1, upper = Inf, tags = c("train", "distill")),
        ParamDbl$new("switch_prob", lower = 0, upper = 1, tags = c("train", "distill")),
        ParamDbl$new("var", lower = 0, upper = Inf, tags = c("train", "distill")),
        ParamDbl$new("alpha", lower = 0, upper = 1, default = 0, tags = c("train", "distill")),
        ParamDbl$new("keep_fraction", lower = 0, upper = 1, default = 0.1, tags = c("train", "distill"))
      )))
      self$param_set$values = c(self$param_set$values, list(min_n = 1e4, switch_prob = 0.5, var = 2, alpha = 0, keep_fraction = 0.1))
      mlr3misc::insert_named(self$param_set$values$callbacks, list(
        "callbacks" = c(self$param_set$values$callbacks, mlr3keras::cb_es(patience=10L)),
        "layer_units" = c(256L, 256L),
        "epochs" = 100L
      ))

    }
  ),
  private = list(

    #' @description
    #' Internal training function, runs the distillation
    #' @param task [`Task`] \cr
    #'   Data the `teacher` was trained on
    #' @return A list with slots 'model', 'history'
    .train = function(task) {
      distill_pars = self$param_set$get_values(tags = "distill")
      pars = self$param_set$get_values(tags = "train")

      model = self$architecture$get_model(task, pars)

      # Either fit directly on data or create a generator and fit from there
      if (!pars$low_memory) {
        task = gen_artificial_data(task, min_n = distill_pars$min_n, keep_fraction = distill_pars$keep_fraction, switch_prob = distill_pars$switch_prob, var = distill_pars$var)
        features = task$data(cols = task$feature_names)
        x = self$architecture$transforms$x(features, pars)
        y = label_artificial_data_classif(task, teacher = self$teacher, prob = distill_pars$probabilities, alpha = distill_pars$alpha)
        y = self$architecture$transforms$y(y, pars = pars, model_loss = model$loss)
        history = invoke(keras::fit,
          object = model,
          x = x,
          y = y,
          epochs = as.integer(pars$epochs),
          class_weight = pars$class_weight,
          batch_size = as.integer(pars$batch_size),
          validation_split = pars$validation_split,
          verbose = as.integer(pars$verbose),
          callbacks = pars$callbacks
        )
      } else {
        generators = make_munge_generators(
          task = task,
          x_transform = function(features) self$architecture$transforms$x(features, pars = pars),
          y_transform = function(target) self$architecture$transforms$y(target, pars = pars, model_loss = model$loss),
          validation_split = pars$validation_split,
          batch_size = pars$batch_size,
          min_n = distill_pars$min_n,  keep_fraction = distill_pars$keep_fraction,
          switch_prob = distill_pars$switch_prob, var = distill_pars$var,
          teacher = self$teacher, prob = distill_pars$probabilities, alpha = distill_pars$alpha
        )
        history = invoke(keras::fit_generator,
          object = model,
          generator = generators$train_gen,
          epochs = as.integer(pars$epochs),
          class_weight = pars$class_weight,
          steps_per_epoch = generators$train_steps,
          validation_data = generators$valid_gen,
          validation_steps = generators$valid_steps,
          verbose = pars$verbose,
          callbacks = pars$callbacks)
      }
      return(list(model = model, history = history))
    }
  )
)