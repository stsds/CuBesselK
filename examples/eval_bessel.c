#include "logbesselk.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

int main() {
    // Calculate array sizes based on the ranges
    int x_count = (0.1 - 0.001) / 0.005 + 1;  // From 0.001 to 0.1 with step 0.005
    int nu_count = (5.0 - 0.001) / 0.25 + 1;  // From 0.001 to 5 with step 0.25

    int total_count = x_count * nu_count;
    int total = total_count * 2;
    double *flattened_list = (double *)malloc(total * sizeof(double));
    double *result = (double *)malloc(total_count * sizeof(double));

    // Generate x and nu values
    int index = 0;
    for (int i = 0; i < x_count; ++i) {
        double x = 0.001 + i * 0.005;  // Step of 0.005 for x
        for (int j = 0; j < nu_count; ++j) {
            double nu = 0.001 + j * 0.25;  // Step of 0.25 for nu
            flattened_list[index++] = x;
            flattened_list[index++] = nu;
        }
    }

    // Separate x and nu arrays for CUDA
    double *x = (double *)malloc(total_count * sizeof(double));
    double *nu = (double *)malloc(total_count * sizeof(double));
    for (int i = 0; i < total_count; ++i) {
        x[i] = flattened_list[2 * i];
        nu[i] = flattened_list[2 * i + 1];
    }

    // Call the CUDA function
    BesselK_CUDA(x, nu, result, total_count);

    // Log results to CSV file
    FILE *f = fopen("besselk.csv", "w");
    fprintf(f, "x,nu,refined_val\n");  // Header
    for (int i = 0; i < total_count; ++i) {
        fprintf(f, "%.15f,%.15f,%.15f\n", x[i], nu[i], log(result[i]));
    }
    fclose(f);

    // Free memory
    free(x);
    free(nu);
    free(result);
    free(flattened_list);



    return 0;
}
