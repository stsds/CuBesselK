/**                
*
* Copyright (c) 2025  King Abdullah University of Science and Technology                                     
* All rights reserved.
*
* CuBesselK is provided by KAUST                                                                             
**/               
/**                
*
* @file array_generation.c                                                                                     
*
* To generate an 2D array of modified Bessel function of the second kind given range.
*
* @author Zipei Geng
* @date 2025-03-17
*                                                                                                                                                                                         
**/ 


#include "logbesselk.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

void print_usage(const char* program_name) {
    printf("Usage: %s [OPTIONS]\n", program_name);
    printf("Options:\n");
    printf("  --nbins N    Set the number of intervals (default: 128)\n");
}

int main(int argc, char *argv[]) {
    int i;
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--nbins") == 0) {
            if (i + 1 < argc) {
                int nbins = atoi(argv[i + 1]);
                if (nbins > 0) {
                    setIntervals(nbins);
                    printf("Using custom intervals: %d\n", nbins);
                } else {
                    printf("Warning: Invalid nbins value. Using default.\n");
                }
                i++; // Skip the next argument (the value)
            } else {
                printf("Error: --nbins requires a value\n");
                print_usage(argv[0]);
                return 1;
            }
        } else {
            printf("Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }
    
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
