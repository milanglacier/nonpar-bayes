#include <Rcpp.h>
using namespace Rcpp;


// [[Rcpp::export]]
NumericMatrix get_sum_adj(NumericMatrix adj_mat) {

    int nrow = adj_mat.nrow();
    int ncol = adj_mat.ncol();

    NumericMatrix adj_sum(ncol, ncol);

    for (int i = 0; i < nrow; i++) {

        for (int j = 0; j < ncol; j++) {

            for (int k = 0; k < ncol; k++) {

                if (adj_mat(i, j) == adj_mat(i, k))

                    adj_sum(j, k) += 1; 


            }

        }

    }

    return adj_sum;
}


// [[Rcpp::export]]
NumericMatrix get_adj_at_a_row(NumericMatrix adj_mat, int row) {


    NumericMatrix adj(adj_mat.ncol(), adj_mat.ncol());

    for (int i = 0; i < adj_mat.ncol(); i++) {

        for (int j = 0; j < adj_mat.ncol(); j++) {

            if (adj_mat(row, i) == adj_mat(row, j)) {

                adj(i, j) = 1;

            }

        }

    }

    return adj;
}


// [[Rcpp::export]]
NumericMatrix expand_adj(NumericMatrix adj) {

    int nrow = adj.nrow();
    NumericMatrix expanded_adj(nrow * nrow, 3);
    int index = 0;

    for (int i = 0; i < nrow; i++) {

        for (int j = 0; j < nrow; j++) {


            expanded_adj(index, 0) = i;
            expanded_adj(index, 1) = j;
            expanded_adj(index, 2) = adj(i, j);
            index++;

        }


    }

    return expanded_adj;
}
