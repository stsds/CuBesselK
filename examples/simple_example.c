/**
 *
 * Copyright (c) 2025  King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * CuBesselK is provided by KAUST
 **/
/**
 *
 * @file simple_example.c
 *
 * A simple example that evaluate modified Bessel function of the second kind using CUDA under various inputs.
 *
 * @author Zipei Geng
 * @date 2025-03-11
 *
 **/


#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "logbesselk.h"

int main() {
    // Number of test cases
    const int total_count = 20;
    
    // Allocate memory for input and output arrays
    double *x = (double *)malloc(total_count * sizeof(double));
    double *nu = (double *)malloc(total_count * sizeof(double));
    double *result = (double *)malloc(total_count * sizeof(double));
    
    // Test case 1: Integer orders with fixed x
    x[0] = 0.5;  nu[0] = 0.0;
    x[1] = 0.5;  nu[1] = 1.0;
    x[2] = 0.5;  nu[2] = 2.0;
    x[3] = 0.5;  nu[3] = 5.0;
    
    // Test case 2: Fractional orders with fixed x
    x[4] = 1.0;  nu[4] = 0.5;
    x[5] = 1.0;  nu[5] = 1.5;
    x[6] = 1.0;  nu[6] = 2.5;
    
    // Test case 3: Small x values with fixed order
    x[7] = 0.001; nu[7] = 1.0;
    x[8] = 0.01;  nu[8] = 1.0;
    x[9] = 0.1;   nu[9] = 1.0;
    
    // Test case 4: Large x values with fixed order
    x[10] = 10.0;  nu[10] = 1.0;
    x[11] = 50.0;  nu[11] = 1.0;
    x[12] = 100.0; nu[12] = 1.0;
    
    // Test case 5: Various combinations
    x[13] = 0.75; nu[13] = 0.25;
    x[14] = 2.5;  nu[14] = 1.33;
    x[15] = 5.0;  nu[15] = 3.5;
    x[16] = 0.05; nu[16] = 4.75;
    
    // Test case 6: Edge cases
    x[17] = 0.001; nu[17] = 0.001; // Both small
    x[18] = 15.0;  nu[18] = 4.0;   // Both large
    x[19] = 0.01;  nu[19] = 3.0;   // Small x, moderate nu
    
    // Call your CUDA function
    BesselK_CUDA(x, nu, result, total_count);
    
    // Print results
    printf("=== BesselK Function Test Results ===\n");
    printf("x\t\tnu\t\tK_nu(x)\n");
    printf("----------------------------------------\n");
    
    // Integer orders
    printf("\n=== Integer Orders (x=0.5) ===\n");
    for(int i = 0; i < 4; i++) {
        printf("%.6f\t%.6f\t%.15e\n", x[i], nu[i], result[i]);
    }
    
    // Fractional orders
    printf("\n=== Fractional Orders (x=1.0) ===\n");
    for(int i = 4; i < 7; i++) {
        printf("%.6f\t%.6f\t%.15e\n", x[i], nu[i], result[i]);
    }
    
    // Small arguments
    printf("\n=== Small Arguments (nu=1.0) ===\n");
    for(int i = 7; i < 10; i++) {
        printf("%.6f\t%.6f\t%.15e\n", x[i], nu[i], result[i]);
    }
    
    // Large arguments
    printf("\n=== Large Arguments (nu=1.0) ===\n");
    for(int i = 10; i < 13; i++) {
        printf("%.6f\t%.6f\t%.15e\n", x[i], nu[i], result[i]);
    }
    
    // Various combinations
    printf("\n=== Various Combinations ===\n");
    for(int i = 13; i < 17; i++) {
        printf("%.6f\t%.6f\t%.15e\n", x[i], nu[i], result[i]);
    }
    
    // Edge cases
    printf("\n=== Edge Cases ===\n");
    for(int i = 17; i < 20; i++) {
        printf("%.6f\t%.6f\t%.15e\n", x[i], nu[i], result[i]);
    }
    
    // Free memory
    free(x);
    free(nu);
    free(result);
    
    return 0;
}
