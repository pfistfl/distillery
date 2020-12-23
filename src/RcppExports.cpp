// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

// swap_factors
IntegerVector swap_factors(const NumericVector ids_to_swap, const IntegerVector munged_col, const NumericVector first_nns, const double var_param);
RcppExport SEXP _distillery_swap_factors(SEXP ids_to_swapSEXP, SEXP munged_colSEXP, SEXP first_nnsSEXP, SEXP var_paramSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const NumericVector >::type ids_to_swap(ids_to_swapSEXP);
    Rcpp::traits::input_parameter< const IntegerVector >::type munged_col(munged_colSEXP);
    Rcpp::traits::input_parameter< const NumericVector >::type first_nns(first_nnsSEXP);
    Rcpp::traits::input_parameter< const double >::type var_param(var_paramSEXP);
    rcpp_result_gen = Rcpp::wrap(swap_factors(ids_to_swap, munged_col, first_nns, var_param));
    return rcpp_result_gen;
END_RCPP
}
// swap_numerics
Rcpp::NumericVector swap_numerics(const NumericVector ids_to_swap, const NumericVector munged_col, const NumericVector first_nns, const double var_param);
RcppExport SEXP _distillery_swap_numerics(SEXP ids_to_swapSEXP, SEXP munged_colSEXP, SEXP first_nnsSEXP, SEXP var_paramSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const NumericVector >::type ids_to_swap(ids_to_swapSEXP);
    Rcpp::traits::input_parameter< const NumericVector >::type munged_col(munged_colSEXP);
    Rcpp::traits::input_parameter< const NumericVector >::type first_nns(first_nnsSEXP);
    Rcpp::traits::input_parameter< const double >::type var_param(var_paramSEXP);
    rcpp_result_gen = Rcpp::wrap(swap_numerics(ids_to_swap, munged_col, first_nns, var_param));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_distillery_swap_factors", (DL_FUNC) &_distillery_swap_factors, 4},
    {"_distillery_swap_numerics", (DL_FUNC) &_distillery_swap_numerics, 4},
    {NULL, NULL, 0}
};

RcppExport void R_init_distillery(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
