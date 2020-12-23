context("rcpp functions to swap values in columns")
test_that("swap_factors", {
  ids_to_swap <- c(1, 3, 6)
  munged_col <- factor(c(1:6))
  nns <- c(2, 3, 4, 6, 6, 1)
  expect_equivalent(
    swap_factors(ids_to_swap, munged_col, nns, 1),
    factor(c(6, 1, 4, 3, 5, 2)))
})

test_that("swap_numerics", {
  ids_to_swap <- c(1, 3, 6)
  munged_col <- c(1:6)
  nns <- c(2, 3, 4, 6, 6, 1)
  expect_equivalent(
    round(swap_numerics(ids_to_swap, munged_col, nns, 100000)),
    c(6, 1, 4, 3, 5, 2))
})

test_that("swapping in cpp does not affect the variables in R environment", {
  ids_to_swap <- c(1, 3, 6)
  munged_col <- c(1:6)
  nns <- c(2, 3, 4, 6, 6, 1)
  swap_numerics(ids_to_swap, munged_col, nns, 100000)
  expect_equivalent(
    munged_col,
    c(1:6))
  munged_col <- factor(c(1:6))
  swap_factors(ids_to_swap, munged_col, nns, 1)
  expect_equivalent(
    munged_col,
    factor(c(1:6)))
})
