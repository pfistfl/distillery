% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/label_data.R
\name{label_artificial_data_regr}
\alias{label_artificial_data_regr}
\title{Label a dataset using a supplied learner.}
\usage{
label_artificial_data_regr(task, teacher, alpha = 0)
}
\arguments{
\item{task}{[`Task`] \cr
Task to create labels for.}

\item{teacher}{[`Learner`] \cr
Trained learner to compress into a neural network.}

\item{alpha}{[`numeric`] \cr
Specifies a convex-combination of labels predicted by the learner (alpha = 0)
and labels of the observation before swapping.}
}
\value{
A [`vector`] containing the response for each observation
}
\description{
Uses the provided learner in order to predict on the provided task's features.
`prob = TRUE` implements Dark Knowledge as implmented by Hinton, 2015.
If 'prob', predicts probabilities, else response.
The mixing parameter 'alpha' corresponds to a trade-off between the predicted target
and the actual target, where 0 only uses predicted targets (default) and 1 only uses the true label.
}
