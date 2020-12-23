#include <Rcpp.h>
using namespace Rcpp;

//' @export
// [[Rcpp::export]]
IntegerVector swap_factors(const NumericVector ids_to_swap,
                           const IntegerVector munged_col,
                           const NumericVector first_nns,
                           const double var_param) {

  IntegerVector munged = clone(munged_col);

  int i, len_ids = ids_to_swap.size();
  int id;
  int munged_entry;
  int first_nn_entry;

  for(i = 0; i < len_ids; i++) {
    id = ids_to_swap[i] - 1;
    munged_entry = munged[id];
    first_nn_entry = munged[first_nns[id] - 1];

    munged[id] = first_nn_entry;
    munged[first_nns[id] - 1] = munged_entry;
  }

  return munged;

}
