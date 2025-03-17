# GPU-Acceleration of Modified Bessel Function of the Second Kind

This is the repo to the manuscript *GPU-Accelerated Modified Bessel Function of the Second Kind for Gaussian Processes* (https://arxiv.org/pdf/2502.00356)

Modified Bessel functions of the second kind are widely used in physics, engineering, spatial statistics and machine learning. Since contemporary scientific applications, including machine learning, rely on GPUs for acceleration, providing robust GPU-hosted implementations of special functions.

Existing implementations of the modified Bessel function of the second kind rely on CPUs and have limited coverage of the full range of values needed in some applications.

This repo present a robust implementation of the modified Bessel function of the second kind on GPU, which utilizes a recursive implementation [(1)](https://dl.acm.org/doi/pdf/10.1145/355921.355928) of series expansion [(2)](https://www.sciencedirect.com/science/article/pii/0021999175900820) and numerical integration [(3)](https://www.sciencedirect.com/science/article/pii/S2352711021001655).

We cover a range of values commonly used in real applications ($x \in [0, 140]$, $\nu \in [0, 20]$), providing high accuracy compared to common libraries like the GNU Scientific Library (GSL) when referenced to Mathematica as the authority.

Our GPU-accelerated approach also demonstrates a 2.68X performance improvement using a single A100 GPU compared to the GSL on 40-core Intel Cascade Lake CPUs.

# Necessary Libraries

gcc (>= 9.4.0), GNU Make (>= 4.2.1), CUDA (>= 11.4)

# Usage

To use this repo, you can either copy the CUDA functions defined in `src/logbesselk.cu` or adding your own program into the `examples/` and compile using `make` command in the main directory. It will create the `build` folder automatically and your can run your self-defined program as executables.

We also provide two examples for users to check it out how to use the defined BesselK function. You can test and run in the `build/` folder. In all example programs, we've implemented command-line functionality to adjust the bin count for numerical integration. Higher bin counts yield more precise calculations, though there's diminishing returns when your application only involves values of ν below 2.5. For reference, the manuscript includes a comparative analysis examining the performance impact of various bin count settings within the Maximum Likelihood Estimation context.

The example code is given as follows:

```
./simple_example --nbins 128
```

Please *NOTE* that some GPU devices are not compatible with the NVCC architecture `sm75`, please edit `NVCCFLAGS` in `Makefile`.

*IMPORTANT NOTICE:* We use series expansion when $x < 0.1$, and numerical integration when $x \leq 0.1$. To avoid the load imbalance carried by if-else branching, we recommend the users to sort the input elements based on which expression is used for each element. For example, if you wish to evaluate $k$ elements that falls into $x < 0.1$, and assuming your block has 1024 threads available. You should allocate $\lfloor k/1024 \rfloor$ blocks solely evaluating series expansion.

# File hierarchy

- `examples` This folder contains two examples to illustrate the usage of the defined function.
- `include` This folder contains header file(s).
- `src` This folder contains the CUDA program of executing BesselK function.
- `Makefile` Contains all details to compile this project.
