#' @title Validation Loss Measure
#'
#' @section Meta Information:
#' * Type: `NA`
#' * Range: \eqn{[0, \infty)}{[0, Inf)}
#' * Minimize: `TRUE`
#' * Required prediction: 'response'
#'
#' @export
MeasureValLoss = R6Class("MeasureValLoss",
  inherit = mlr3::Measure,
  public = list(
    #' @description
    #' Creates a new measure that measures a keras learner's validation loss.
    #' @param id Id of the measure
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(id = "val_loss") {
      super$initialize(
        id = id,
        task_type = NA_character_,
        predict_type = "response",
        range = c(0, Inf),
        minimize = TRUE,
        properties = "requires_learner",
        man = "distillery::measure_val_loss"
      )
    }
  ),

  private = list(
    .score = function(prediction, learner, ...) {
      losses = learner$model$history$metrics$val_loss
      losses[length(losses)]
    }
  )
)
