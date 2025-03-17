/**                                                                                                                                                                                        
*
* Copyright (c) 2025  King Abdullah University of Science and Technology                                     
* All rights reserved.
*
* CuBesselK is provided by KAUST                                                                             
**/
/**                
*
* @file logbesselk.cu                                                                                   
*
* CUDA device and global function defined to evaluate modified Bessel function of the second kind on CUDA device.
*
* @author Zipei Geng
* @date 2025-03-17
*                                                                                                                                                                                         
**/


#include "logbesselk.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// Host variable
int intervals = 128;

// A device constant that will store the value
__constant__ int d_intervals;

// Function to update the device constant
void setIntervals(int nbins) {
    intervals = nbins;
    cudaMemcpyToSymbol(d_intervals, &intervals, sizeof(int));
}

// Copy the initial value to the device
void initBesselK() {
    cudaMemcpyToSymbol(d_intervals, &intervals, sizeof(int));
}

__constant__ double g1_dat[14] = {
  -1.14516408366268311786898152867,
   0.00636085311347084238122955495,
   0.00186245193007206848934643657,
   0.000152833085873453507081227824,
   0.000017017464011802038795324732,
  -6.4597502923347254354668326451e-07,
  -5.1819848432519380894104312968e-08,
   4.5189092894858183051123180797e-10,
   3.2433227371020873043666259180e-11,
   6.8309434024947522875432400828e-13,
   2.8353502755172101513119628130e-14,
  -7.9883905769323592875638087541e-16,
  -3.3726677300771949833341213457e-17,
  -3.6586334809210520744054437104e-20
};

__constant__ cheb_series g1_cs = {
    g1_dat,
    -1, 1,
    13
};

__constant__ double g2_dat[15] = 
{
  1.882645524949671835019616975350,
 -0.077490658396167518329547945212,  
 -0.018256714847324929419579340950,
  0.0006338030209074895795923971731,
  0.0000762290543508729021194461175,
 -9.5501647561720443519853993526e-07,
 -8.8927268107886351912431512955e-08,
 -1.9521334772319613740511880132e-09,
 -9.4003052735885162111769579771e-11,
  4.6875133849532393179290879101e-12,
  2.2658535746925759582447545145e-13,
 -1.1725509698488015111878735251e-15,
 -7.0441338200245222530843155877e-17,
 -2.4377878310107693650659740228e-18,
 -7.5225243218253901727164675011e-20
};

__constant__ cheb_series g2_cs = {
    g2_dat,
    -1, 1,
    14
};

__device__ void cheb_eval_cuda(const cheb_series* cs,
               const double x, double *result)
{
    int j;
    double d = 0.0;
    double dd = 0.0;

    double y = (2.0 * x - cs->a - cs->b) / (cs->b - cs->a);
    double y2 = 2.0 * y;

    for (j = cs->order; j >= 1; j--) {
        double temp = d;
        d = y2 * d - dd + cs->c[j];
        dd = temp;
    }

    {
        // double temp = d;
        d = y * d - dd + 0.5 * cs->c[0];
    }

    *result = d;

} 

__device__ void temme_gamma(const double nu, double *g_1pnu, double *g_1mnu,
                            double *g1, double * g2)
{
    double anu = fabs(nu);
    double x = 4.0 * anu - 1.0;
    double r_g1, r_g2;

    cheb_eval_cuda(&g1_cs, x, &r_g1);
    cheb_eval_cuda(&g2_cs, x, &r_g2);

    *g1 = r_g1;
    *g2 = r_g2;
    *g_1mnu = 1.0 / (r_g2 + nu * r_g1);
    *g_1pnu = 1.0 / (r_g2 - nu * r_g1);
}

__device__ void besselk_scaled_temme(const double nu, const double x,
                                      double * K_nu, double * K_nup1, double * Kp_nu)
{
    const int    max_iter  = 15000;
    const double half_x    = 0.5 * x;
    const double ln_half_x = log(half_x);
    const double half_x_nu = exp(nu * ln_half_x);
    const double pi_nu     = M_PI * nu;
    const double sigma     = -nu * ln_half_x;
    const double sinrat    = (fabs(pi_nu) < DBL_EPSILON ? 1.0 : pi_nu/sin(pi_nu));
    const double sinhrat   = (fabs(sigma) < DBL_EPSILON ? 1.0 : sinh(sigma)/sigma);
    const double ex        = exp(x);
    double sum0, sum1;
    double fk, pk, qk, hk, ck;
    int k = 0;

    double g_1pnu, g_1mnu, g1, g2;
    temme_gamma(nu, &g_1pnu, &g_1mnu, &g1, &g2);

    fk = sinrat * (cosh(sigma) * g1 - sinhrat * ln_half_x * g2);
    pk = 0.5 / half_x_nu * g_1pnu;
    qk = 0.5 * half_x_nu * g_1mnu;
    hk = pk;
    ck = 1.0;
    sum0 = fk;
    sum1 = hk;

    while(k < max_iter) {
        double del0;
        double del1;
        k++;
        fk  = (k * fk + pk + qk) / (k * k - nu * nu);
        ck *= half_x * half_x / k;
        pk /= (k - nu);
        qk /= (k + nu);
        hk  = -k * fk + pk;
        del0 = ck * fk;
        del1 = ck * hk;
        sum0 += del0;
        sum1 += del1;
        if(fabs(del0) < 0.5 * fabs(sum0) * DBL_EPSILON) break;
    }
  
    *K_nu   = sum0 * ex;
    *K_nup1 = sum1 * 2.0 / x * ex;
    *Kp_nu  = - (*K_nup1) + nu / x * (*K_nu);
}

// Start the Takekawa's implementation (adapted version)

__device__ double f(double t, double v, double x) {
    double ct = cosh(t);
    double cvt = cosh(v * t);
    return log(cvt) - x * ct;
}

__device__ double modified_besselk(double nu, double x) {

    if (x <= 0) return INFINITY;
    
    if (nu < 0) nu = -nu; 
    
    if (x <= 0.1) {
        int N = (int)(nu + 0.5);
        double mu = nu - N;      /* -1/2 <= mu <= 1/2 */
        double K_mu, K_mup1, Kp_mu;
        double K_nu, K_nup1, K_num1;
        int n, e10 = 0;

        besselk_scaled_temme(mu, x, &K_mu, &K_mup1, &Kp_mu);

        /* recurse forward to obtain K_num1, K_nu */
        K_nu   = K_mu;
        K_nup1 = K_mup1;

        for(n = 0; n < N; n++) {
            K_num1 = K_nu;
            K_nu   = K_nup1;
            /* rescale the recurrence to avoid overflow */
            if (fabs(K_nu) > SQRT_DBL_MAX) {
                double p = floor(log(fabs(K_nu)) / M_LN10);
                double factor = pow(10.0, p);
                K_num1 /= factor;
                K_nu /= factor;
                e10 += p;
            };
            K_nup1 = 2.0 * (mu + n + 1) / x * K_nu + K_num1;
        }
        return K_nu * exp(-x);
    } 
    else {
        const double h = 9.0 / d_intervals; // upper bound = 9, lower bound = 0
        double max_term = -INFINITY;
        double sum = 0.0;

        // First pass: find the maximum term
        for (int m = 0; m <= d_intervals; m++) {
            double t_m = m * h;
            double c_m = (m == 0 || m == d_intervals) ? 0.5 : 1.0;
            double g_m = f(t_m, nu, x);
            double term = log(c_m) + g_m;
            max_term = fmax(max_term, term);
        }

        // Second pass: compute the sum using the log-sum-exp trick
        for (int m = 0; m <= d_intervals; m++) {
            double t_m = m * h;
            double c_m = (m == 0 || m == d_intervals) ? 0.5 : 1.0;
            double g_m = f(t_m, nu, x);
            double term = log(c_m) + g_m;
            sum += exp(term - max_term);
        }

        double res = log(h) + max_term + log(sum);
        return exp(res);
    }
}

__global__ void modified_besselkKernel(double *x, double *v, double *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = modified_besselk(v[idx], x[idx]);
    }
}

extern "C" void BesselK_CUDA(double *host_x, double *host_v,
                             double *host_result, int n) {

    static bool initialized = false;
    if (!initialized) {
        initBesselK();
        initialized = true;
    }
    double *dev_x, *dev_v, *dev_result;
    
    cudaMalloc((void**)&dev_x, n * sizeof(double));
    cudaMalloc((void**)&dev_v, n * sizeof(double));
    cudaMalloc((void**)&dev_result, n * sizeof(double));
    
    cudaMemcpy(dev_x, host_x, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_v, host_v, n * sizeof(double), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    modified_besselkKernel<<<blocksPerGrid, threadsPerBlock>>>(dev_x, dev_v, dev_result, n);
    
    cudaMemcpy(host_result, dev_result, n * sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaFree(dev_x);
    cudaFree(dev_v);
    cudaFree(dev_result);
}
