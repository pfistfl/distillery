context("munge works")

test_that("munge iris", {
  t = tsk("iris")
  dt = munge_task(t)
  out = pmap_lgl(list(t$data(), dt), function(x,y) {
    class(x) == class(y)
  })
  expect_true(all(out))
})

test_that("munge credit", {
  t = tsk("german_credit")
  dt = munge_task(t)
  out = pmap_lgl(list(t$data(), dt), function(x,y) {
    class(x)[1] == class(y)[1]
  })
  expect_true(all(out))
})
