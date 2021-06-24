/**
 * Some part of the source code is from the SHOC benchmark suite.
 *
 * Original authors:
 * Collin McCurdy, September 08, 2009
 * Vasily Volkov
 *
 * Here is the original copyright notice and a note from the original implementation:
 * This code uses algorithm described in:
 * "Fitting FFT onto G80 Architecture". Vasily Volkov and Brian Kazian, UC Berkeley CS258 project report. May 2008.
 *
 * Written by Vasily Volkov.
 * Copyright (c) 2008-2009, The Regents of the University of California.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 *    - Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimer.
 *    - Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer
 *        in the documentation and/or other materials provided with the
 *        distribution.
 *    - Neither the name of the University of California, Berkeley nor the
 *        names of its contributors may be used to endorse or promote
 *        products derived from this software without specific prior
 *        written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <boost/math/special_functions/next.hpp>
#include <boost/program_options.hpp>
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <sys/mman.h>
#include <tuple>

#include "../device_hopper/core.h"
#include "shoc_src/fftlib.h"

using namespace device_hopper;

#define REPOSITORY_PATH std::string(std::getenv("PLASTICITY_ROOT"))
#define PREFERRED_DEVICE CPU

#ifndef M_PI
# define M_PI 3.14159265358979323846f
#endif

#ifndef M_SQRT1_2
# define M_SQRT1_2      0.70710678118654752440f
#endif

#define exp_1_8   (float2){  1.0f, -1.0f }//requires post-multiply by 1/sqrt(2)
#define exp_1_4   (float2){  0.0f, -1.0f }
#define exp_3_8   (float2){ -1.0f, -1.0f }//requires post-multiply by 1/sqrt(2)

#define iexp_1_8   (float2){  1.0f, 1.0f }//requires post-multiply by 1/sqrt(2)
#define iexp_1_4   (float2){  0.0f, 1.0f }
#define iexp_3_8   (float2){ -1.0f, 1.0f }//requires post-multiply by 1/sqrt(2)


inline void globalLoads8(float2 *data, float2 *work, int stride) {
    for (int i = 0; i < 8; i++)
        data[i] = work[i * stride];
}


inline void globalStores8(float2 *data, float2 *work, int stride) {
    int reversed[] = {0, 4, 2, 6, 1, 5, 3, 7};

//#pragma unroll
    for (int i = 0; i < 8; i++) {
        float2 d = data[reversed[i]];
        work[i * stride] = d;
    }
}


inline void storex8(float2 *a, float *smem, int sx) {
    int reversed[] = {0, 4, 2, 6, 1, 5, 3, 7};

//#pragma unroll
    for (int i = 0; i < 8; i++)
        smem[i * sx] = a[reversed[i]].x;
}

inline void storey8(float2 *a, float *smem, int sx) {
    int reversed[] = {0, 4, 2, 6, 1, 5, 3, 7};

//#pragma unroll
    for (int i = 0; i < 8; i++)
        smem[i * sx] = a[reversed[i]].y;
}


inline void loadx8(float2 *a, float *smem, int sx) {
    for (int i = 0; i < 8; i++)
        a[i].x = smem[i * sx];
}

inline void loady8(float2 *a, float *smem, int sx) {
    for (int i = 0; i < 8; i++)
        a[i].y = smem[i * sx];
}


#define transpose(a, s, ds, l, dl, sync)                              \
{                                                                       \
    storex8( a, s, ds );  if( (sync)&8 ) device_hopper::batch_barrier(); \
    loadx8 ( a, l, dl );  if( (sync)&4 ) device_hopper::batch_barrier();  \
    storey8( a, s, ds );  if( (sync)&2 ) device_hopper::batch_barrier();  \
    loady8 ( a, l, dl );  if( (sync)&1 ) device_hopper::batch_barrier();  \
}

inline float2 exp_i(float phi) {
//#ifdef USE_NATIVE
//    return (float2)( native_cos(phi), native_sin(phi) );
//#else
    return (float2) {cos(phi), sin(phi)};
//#endif
}

inline float2 cmplx_mul(float2 a, float2 b) { return (float2) {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x}; }

inline float2 cm_fl_mul(float2 a, float b) { return (float2) {b * a.x, b * a.y}; }

inline float2 cmplx_add(float2 a, float2 b) { return (float2) {a.x + b.x, a.y + b.y}; }

inline float2 cmplx_sub(float2 a, float2 b) { return (float2) {a.x - b.x, a.y - b.y}; }

//#define IFFT2 FFT2

#define twiddle8(a, i, n)                                              \
{                                                                       \
    int reversed8[] = {0,4,2,6,1,5,3,7};                                \
    for( int j = 1; j < 8; j++ ){                                       \
        a[j] = cmplx_mul( a[j],exp_i((-2*M_PI*reversed8[j]/(n))*(i)) ); \
    }                                                                   \
}

#define FFT2(a0, a1)                            \
{                                               \
    float2 c0 = *a0;                           \
    *a0 = cmplx_add(c0,*a1);                    \
    *a1 = cmplx_sub(c0,*a1);                    \
}

#define FFT4(a0, a1, a2, a3)                    \
{                                               \
    FFT2( a0, a2 );                             \
    FFT2( a1, a3 );                             \
    *a3 = cmplx_mul(*a3,exp_1_4);               \
    FFT2( a0, a1 );                             \
    FFT2( a2, a3 );                             \
}

#define FFT8(a)                                                 \
{                                                               \
    FFT2( &a[0], &a[4] );                                       \
    FFT2( &a[1], &a[5] );                                       \
    FFT2( &a[2], &a[6] );                                       \
    FFT2( &a[3], &a[7] );                                       \
                                                                \
    a[5] = cm_fl_mul( cmplx_mul(a[5],exp_1_8) , M_SQRT1_2 );    \
    a[6] =  cmplx_mul( a[6] , exp_1_4);                         \
    a[7] = cm_fl_mul( cmplx_mul(a[7],exp_3_8) , M_SQRT1_2 );    \
                                                                \
    FFT4( &a[0], &a[1], &a[2], &a[3] );                         \
    FFT4( &a[4], &a[5], &a[6], &a[7] );                         \
}

#define itwiddle8(a, i, n)                                            \
{                                                                       \
    int reversed8[] = {0,4,2,6,1,5,3,7};                                \
    for( int j = 1; j < 8; j++ )                                        \
        a[j] = cmplx_mul(a[j] , exp_i((2*M_PI*reversed8[j]/(n))*(i)) ); \
}

#define IFFT4(a0, a1, a2, a3)                 \
{                                               \
    FFT2( a0, a2 );                            \
    FFT2( a1, a3 );                            \
    *a3 = cmplx_mul(*a3 , iexp_1_4);            \
    FFT2( a0, a1 );                            \
    FFT2( a2, a3);                             \
}

#define IFFT8(a)                                              \
{                                                               \
    FFT2( &a[0], &a[4] );                                      \
    FFT2( &a[1], &a[5] );                                      \
    FFT2( &a[2], &a[6] );                                      \
    FFT2( &a[3], &a[7] );                                      \
                                                                \
    a[5] = cm_fl_mul( cmplx_mul(a[5],iexp_1_8) , M_SQRT1_2 );   \
    a[6] = cmplx_mul( a[6] , iexp_1_4);                         \
    a[7] = cm_fl_mul( cmplx_mul(a[7],iexp_3_8) , M_SQRT1_2 );   \
                                                                \
    IFFT4( &a[0], &a[1], &a[2], &a[3] );                        \
    IFFT4( &a[4], &a[5], &a[6], &a[7] );                        \
}

typedef struct cplxflt {
    float x;
    float y;
} cplxflt;

void populate_input_buffer(cplxflt *buffer_in_out, size_t nftts) {
    size_t half_n_cmplx = (nftts * 512) / 2;
    for (size_t i = 0; i < half_n_cmplx; ++i) {
        buffer_in_out[i].x = (rand() / (float) RAND_MAX) * 2 - 1;
        buffer_in_out[i].y = (rand() / (float) RAND_MAX) * 2 - 1;
        buffer_in_out[i + half_n_cmplx].x = buffer_in_out[i].x;
        buffer_in_out[i + half_n_cmplx].y = buffer_in_out[i].y;
    }
}

void apply_fft(cplxflt *b, unsigned long b_size, size_t n_ffts, bool ifft=false) {
    cl_uint num_platforms = 0;
    cl_platform_id platform[2];
    cl_device_id device;
    cl_int err = clGetPlatformIDs(2, platform, &num_platforms);
    err |= clGetDeviceIDs(
            platform[0],
            CL_DEVICE_TYPE_CPU,
            1,
            &device,
            NULL);
    if (err == -1 && num_platforms > 1) {
        std::cout << "INFO: First platform does not offer the target device. "
                     "Trying to use the second platform..." << std::endl;
        err = clGetDeviceIDs(
                platform[1],
                CL_DEVICE_TYPE_CPU,
                1,
                &device,
                NULL);
        if (err) plasticity::cl_utils::cl_check_return_code(err, "Could not find the CPU");
    }

    // Create the OCL context handle
    cl_context cl_ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    plasticity::cl_utils::cl_check_return_code(err, "Could not initialise the OpenCL context");

    // Load and compile the .cl file
    std::string cl_file_path = REPOSITORY_PATH + "/library/" + "/plasticity/opencl_files/shoc_fft.cl";
    FILE *program_handle = fopen(cl_file_path.c_str(), "r");
    if (program_handle == NULL) {
        std::cerr << "Error: Could not open the OpenCL source code file" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    err |= fseek(program_handle, 0, SEEK_END);
    size_t program_size = ftell(program_handle);
    rewind(program_handle);
    char *program_buffer = (char *) malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);
    //std::cout << program_buffer << std::endl;
    cl_program program = clCreateProgramWithSource(
            cl_ctx,
            1,
            (const char **) &program_buffer,
            &program_size,
            &err);
    plasticity::cl_utils::cl_check_return_code(err, "Could create program handle");
    free(program_buffer);
    std::string flags = "";
    flags += "-cl-fast-relaxed-math ";
    flags += "-cl-mad-enable ";
    flags += " -DSINGLE_PRECISION";

    err |= clBuildProgram(program, 1, &device, flags.c_str(), NULL, NULL);
    if (err) {
        std::cerr << "Program building failed: ERROR " << err << std::endl;
        // Print warnings and errors from compilation
        static char log[65536];
        memset(log, 0, sizeof(log));
        err = clGetProgramBuildInfo(program,
                                    device,
                                    CL_PROGRAM_BUILD_LOG,
                                    sizeof(log) - 1,
                                    log,
                                    NULL);
        printf("-----OpenCL Compiler Output-----\n");
        if (strstr(log, "warning:") || strstr(log, "error:"))
            printf("<<<<\n%s\n>>>>\n", log);
        printf("--------------------------------\n");
        std::exit(EXIT_FAILURE);
    }
    cl_kernel fftKrnl;
    if (ifft) {
        fftKrnl= clCreateKernel(program, "ifft1D_512_original", &err);
    } else {
        fftKrnl= clCreateKernel(program, "fft1D_512_original", &err);
    }
    plasticity::cl_utils::cl_check_return_code(err, "Could not create an OpenCL kernel handle");
    cl_command_queue queue = clCreateCommandQueue(cl_ctx, device, 0, &err);
    plasticity::cl_utils::cl_check_return_code(err, "Could not create an OpenCL queue");

    // Copy the output of the FFT to the device and perform the inverse FFT.
    cl_mem device_buf = allocHostBuffer(b, b_size, cl_ctx);
    err = transform(device_buf, n_ffts, fftKrnl, queue);
    plasticity::cl_utils::cl_check_return_code(err, "Could not launch kernel");
    err = clFinish(queue);
    plasticity::cl_utils::cl_check_return_code(err, "Could not run kernel");
}

bool verify_results(cplxflt *b, cplxflt *reference, unsigned long b_size, size_t n_ffts) {
    // Check if the results of the inverse FFT are close enough to the original input of the FFT.
    for (int i = 0; i < n_ffts / 2; ++i) {
#if defined(FFT)
        float x_diff = fabs(host_buffer_for_checks[i].x - reference[i].x);
        float y_diff = fabs(host_buffer_for_checks[i].y - reference[i].y);
#else
        float x_diff = fabs(b[i].x - reference[i].x);
        float y_diff = fabs(b[i].y - reference[i].y);
#endif
        if (x_diff > 0.0001 || y_diff > 0.0001 || std::isnan(b[i].x) || std::isnan(b[i].y)) {
            std::cout << "Error at: " << i << std::endl;
            std::cout << "Expected: " << reference[i].x << " " << reference[i].y << std::endl;
#if defined(FFT)
            std::cout << "Is: " << host_buffer_for_checks[i].x << " " << host_buffer_for_checks[i].y << std::endl;
#else
            std::cout << "Is: " << b[i].x << " " << b[i].y << std::endl;
#endif

            return false;
        }
    }
    return true;
}

DEVICE_HOPPER_MAIN(int argc, char *argv[]) {
    DEVICE_HOPPER_SETUP
    const size_t block_size = 64;

    // Parse CLI parameters
    boost::program_options::options_description desc("Options");
    desc.add_options()("problem-size", boost::program_options::value<long>(), "Sample count");
    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    size_t n_ffts = 0;
    if (vm.count("problem-size") == 0) {
        std::cerr << "Error: Problem size is missing." << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        // This parameter is passed to the application by the Python scripts that automate the experiments.
        n_ffts = vm["problem-size"].as<long>();
    }
    const size_t buffer_size = n_ffts * 512u;

    // Allocate buffers
    cplxflt *buffer_in_out = (cplxflt *) device_hopper::aligned_malloc(4096, buffer_size, sizeof(cplxflt));

    // Init input data
    populate_input_buffer(buffer_in_out, n_ffts);

    // Create a copy of the input buffer to compute the reference results later
    cplxflt *reference_buffer = (cplxflt *) std::malloc(buffer_size * sizeof(cplxflt));
    std::memcpy(reference_buffer, buffer_in_out, buffer_size * sizeof(cplxflt));

    // Apply FFT
    apply_fft(buffer_in_out, buffer_size * sizeof(cplxflt), n_ffts, false);

    parallel_for pf(0, n_ffts * block_size, [=]DEVICE_HOPPER_LAMBDA() {
        const int tid = GET_ITERATION_WITHIN_BATCH();
        const int bid = GET_BATCH_ID() * 512 + tid;
        int hi = tid >> 3;
        int lo = tid & 7;
        float2 data[8];
        __attribute__((device_hopper_batch_shared)) float smem[8 * 8 * 9]; //__local

        float2 *work = (float2 *) buffer_in_out;

        // starting index of data to/from global memory
        work = work + bid;
        //out = out + blockIdx;
        globalLoads8(data, work, 64); // coalesced global reads

        IFFT8(data);
        itwiddle8(data, tid, 512);
        transpose(data, &smem[hi * 8 + lo], 66, &smem[lo * 66 + hi], 8, 0xf);
        IFFT8(data);
        itwiddle8(data, hi, 64);
        transpose(data, &smem[hi * 8 + lo], 8 * 9, &smem[hi * 8 * 9 + lo], 8, 0xE);
        IFFT8(data);

        for(int i=0; i<8; i++) {
            data[i].x = data[i].x/512.0f;
            data[i].y = data[i].y/512.0f;
        }

        globalStores8(data, work, 64);
    });
    pf.add_buffer_access_patterns(
        device_hopper::buf(buffer_in_out, direction::IN_OUT, pattern::SUCCESSIVE_SUBSECTIONS(pf.batch_size * 8)));
    // Set optional tuning parameters and call run()
    pf.opt_set_simple_indices(true).opt_set_batch_size(64).run();

    if (!verify_results(buffer_in_out, reference_buffer, buffer_size * sizeof(cplxflt), n_ffts)) {
        std::cout << "Error: The results are incorrect" << std::endl;
        return EXIT_FAILURE;
    } else {
        std::cout << "Info: The results are correct" << std::endl;
        return EXIT_SUCCESS;
    }
}