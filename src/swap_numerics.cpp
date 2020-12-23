#include <Rcpp.h>
#include <cmath>
using namespace Rcpp;

//' @export
// [[Rcpp::export]]
Rcpp::NumericVector swap_numerics(
    const NumericVector ids_to_swap,
    const NumericVector munged_col,
    const NumericVector first_nns,
    const double var_param) {

  NumericVector munged = clone(munged_col);

  int i, len_ids = ids_to_swap.size();
  int id;
  double munged_entry;
  double first_nn_entry;

  for(i = 0; i < len_ids; i++) {
    id = ids_to_swap[i] - 1;
    munged_entry = munged[id];
    first_nn_entry = munged[first_nns[id] - 1];

    double dist = abs(munged_entry - first_nn_entry);
    double sigma = (dist + 0.0001) / var_param ;

    munged[id] = Rcpp::rnorm(1, first_nn_entry, sigma)[0];
    munged[first_nns[id] - 1] = Rcpp::rnorm(1, munged_entry, sigma)[0];
  }
  return munged;
}
