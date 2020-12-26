# Append to reflections
register_mlr3keras = function() {
  x = utils::getFromNamespace("keras_reflections", ns = "mlr3keras")
  x$loss$classif = unique(c(x$loss$classif, "mean_squared_error"))
  mlr_measures$add("val_loss", MeasureValLoss, id = "val_loss", stages = "train")
}

.onLoad = function(libname, pkgname) {  # nocov start
  register_mlr3keras()
  setHook(packageEvent("mlr3keras", "onLoad"), function(...) register_mlr3keras(), action = "append")
  backports::import(pkgname)
}  # nocov end

.onUnload = function(libpath) { # nocov start
   event = packageEvent("mlr3keras", "onLoad")
   hooks = getHook(event)
   pkgname = vapply(hooks[-1], function(x) environment(x)$pkgname, NA_character_)
   setHook(event, hooks[pkgname != "distillery"], action = "replace")
} # nocov end
