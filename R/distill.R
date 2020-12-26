#' @export
distill = function(learner, task, measure = NULL, budget = 5L) {
  assert_task(task)
  assert_learner(learner, task = task)

  if (inherits(task, "TaskRegr")) {
   lrn = LearnerRegrKerasDistill$new(learner)
   if (is.null(measure)) measure = mlr3::msr("val_loss")
  } else {
    lrn = LearnerClassifKerasDistill$new(learner)
    if (is.null(measure)) measure = mlr3::msr("val_loss")
  }

  assert_measure(measure, task, learner)

  design = get_distill_design(lrn)
  tuner = mlr3tuning::tnr("design_points", design = design)
  # Hack ParamUty into param_classes
  tuner$param_classes = c(tuner$param_classes, "ParamUty")
  tuner$`.__enclos_env__`$private$.optimizer$param_classes = tuner$param_classes

  search_space = ParamSet$new(list(
    ParamUty$new("layer_units"),
    ParamInt$new("min_n", lower = 1, upper = Inf),
    ParamDbl$new("alpha", lower = 0, upper = 1)
  ))

  at = mlr3tuning::AutoTuner$new(
    lrn,
    mlr3::rsmp("holdout"),
    measure,
    mlr3tuning::trm("evals", n_evals = budget),
    tuner = tuner,
    search_space = search_space,
    store_models = TRUE
  )
  at$train(task)
  return(at)
}

get_distill_design = function(lrn) {
  dt = data.table(
    min_n = c(1e4, 5*1e4, 1e5, 1e5, 3*1e5),
    alpha = c(0, 0.01, 0, 0.01, 0),
    layer_units = list(c(32L, 32L), c(64L, 64L), c(128L, 64L), c(128L, 128L), c(256L, 256L))
  )
}
