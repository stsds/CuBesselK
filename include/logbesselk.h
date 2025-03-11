/**                
 *
 * Copyright (c) 2025  King Abdullah University of Science and Technology                                     
 * All rights reserved.
 *
 * CuBesselK is provided by KAUST                                                                             
 **/               
/**                
 *
 * @file logbesselk.h                                                                                    
 *
 * Header files to be referenced.
 *
 * @author Zipei Geng
 * @date 2025-03-11
 *                                                                                                                                                                                         
 **/ 


#ifndef CUDA_BESSEL_H
#define CUDA_BESSEL_H

#define DBL_EPSILON        2.2204460492503131e-16
#define SQRT_DBL_MAX       1.3407807929942596e+154


// define the cheb series struct
struct cheb_series_struct {
    double* c;     /* coefficients                     */
    double a;      /* lower interval point             */
    double b;      /* upper interval point             */
    int order;     /* order of the expansion           */
};
typedef struct cheb_series_struct cheb_series;


#ifdef __cplusplus
extern "C" {
#endif

// Function prototype for the C wrapper
void BesselK_CUDA(double *host_x, double *host_v,
                             double *host_result, int n);

#ifdef __cplusplus
}
#endif

#endif // CUDA_BESSEL_H
