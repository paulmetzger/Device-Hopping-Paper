/**
 * Some part of the source code is from the SHOC benchmark suite.
 *
 * Original authors:
 * Anthony Danalis
 * Kyle Spafford
 * Jeremy Meredith
 *
 * This is the original copyright notice:
 * Code derived from work done by the authors quoted in the original header below:
 * (c) January 24, 2008 Vasily Volkov @ UC Berkeley
 * Other credits:
 * - Paul Leventis @ Altera Corp. for prefetching and -maxrregcount techniques
 * - many thanks to Wladimir J. van der Laan @ the University of Groningen
 * for his cubin disassembler (http://www.cs.rug.nl/~wladimir/decuda/)
 */

#include <boost/crc.hpp>
#include <boost/math/special_functions/next.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <tuple>

#include "../device_hopper/core.h"


#define REPOSITORY_PATH std::string(std::getenv("PLASTICITY_ROOT"))
#define PREFERRED_DEVICE GPU

#define SAXPY( _A_, _BS_ , _C_) do{ \
    _C_[0] += _A_ * _BS_[0]; \
    _C_[1] += _A_ * _BS_[1]; \
    _C_[2] += _A_ * _BS_[2]; \
    _C_[3] += _A_ * _BS_[3]; \
    _C_[4] += _A_ * _BS_[4]; \
    _C_[5] += _A_ * _BS_[5]; \
    _C_[6] += _A_ * _BS_[6]; \
    _C_[7] += _A_ * _BS_[7]; \
    _C_[8] += _A_ * _BS_[8]; \
    _C_[9] += _A_ * _BS_[9]; \
    _C_[10] += _A_ * _BS_[10]; \
    _C_[11] += _A_ * _BS_[11]; \
    _C_[12] += _A_ * _BS_[12]; \
    _C_[13] += _A_ * _BS_[13]; \
    _C_[14] += _A_ * _BS_[14]; \
    _C_[15] += _A_ * _BS_[15]; \
    }while(0)

using namespace device_hopper;

void gen_data(float *a_in, float *b_in, float *c_out, size_t n) {
    //srand(123);
    int position = 0;
    for(uint64_t i = 0; i < n * n; i++) {
        if (i == position) {
            a_in[i] = 1;
            position += (n + 1);
        } else a_in[i] = 0;
        //a_in[i]  = 1; //(rand() % 100) / 100.0f;//(i % 3) / 3.0f;
        b_in[i]  = i;//(rand() % 100) / 100.0f;//((i + 1) % 3) / 3.0f;
        c_out[i] = 0.0f;
    }
}

DEVICE_HOPPER_MAIN(int argc, char* argv[]) {
    DEVICE_HOPPER_SETUP

    // Parse CLI parameters
    boost::program_options::options_description desc("Options");
    desc.add_options() ("problem-size", boost::program_options::value<long>(), "Sample count");
    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    size_t n = 0;
    if (vm.count("problem-size") == 0) {
        std::cerr << "Error: Problem size is missing." << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        // This parameter is passed to the application by the Python scripts that automate the experiments.
        n = vm["problem-size"].as<long>();
    }

#if defined(SOURCE_TO_SOURCE_TRANSLATED)
    // This sets the number of blocks in a slice as slice size as expected by the parallel for.
    // The hand implementation uses the number of iteration points in a slice as slice size.
    // To keep the database tables and configs consistent we continue to use the old slice sizes in terms of iteration points
    // in the configs and databases but convert them then here in the source code.
    context.sched.set_cpu_slice_sizes(context.sched.get_cpu_x_slice_size() / 64, context.sched.get_cpu_y_slice_size() / 16);
    context.sched.set_gpu_slice_sizes(context.sched.get_gpu_x_slice_size() / 64, context.sched.get_gpu_y_slice_size() / 16);
#endif

    // Allocate buffer
    size_t data_type_size = sizeof(float);
    size_t buffer_elements = n * n;
    float *a_in  = (float *) device_hopper::aligned_malloc(4096, buffer_elements, data_type_size);
    float *b_in  = (float *) device_hopper::aligned_malloc(4096, buffer_elements, data_type_size);
    float *c_out = (float *) device_hopper::aligned_malloc(4096, buffer_elements, data_type_size);

    // Prepare scalar parameters
    int lda, ldb, ldc;
    lda = ldb = ldc = n;
    float alpha = 1.0f;
    float beta = 0.0f;
    int k = n;

    // Generate Input data
    gen_data(a_in, b_in, c_out, n);

    // Create parallel for
    parallel_for pf(std::make_tuple(0, 0), std::make_tuple(n/4, n/4), [=]DEVICE_HOPPER_LAMBDA() mutable {
        const int inx = GET_2D_ITERATION_WITHIN_BATCH_X;
        const int iny = GET_2D_ITERATION_WITHIN_BATCH_Y;
        //const int ibx = get_group_id(0) * 64;
        //const int iby = get_group_id(1) * 16;
        //const int ibx = (get_global_id(0) / 16) * 64;
        //const int iby = (get_global_id(1) / 4) * 16;
        const int ibx = (GET_2D_ITERATION_X / 16) * 64;
        const int iby = (GET_2D_ITERATION_Y / 4) * 16;
        const int id = inx + iny * 16;
        int i, ii; //, counter=0;

        a_in += ibx + id;
        b_in += inx + (iby+iny) * ldb;
        c_out += ibx + id  + (iby*ldc);

        float c[16];
        for(i=0; i<16; ++i) {
            c[i] = 0.0;
        }

        __attribute__((device_hopper_batch_shared)) float bs[16][17];
        for (int counter = 0; counter < k; counter += 16) {
            float a[4];
            for(ii=0; ii<4; ++ii) {
                a[ii] = a_in[ii*lda];
            }

            bs[inx][iny]    = b_in[0*ldb];
            bs[inx][iny+4]  = b_in[4*ldb];
            bs[inx][iny+8]  = b_in[8*ldb];
            bs[inx][iny+12] = b_in[12*ldb];
            device_hopper::batch_barrier();

            a_in += 4*lda;

            SAXPY( a[0], bs[0], c );	a[0] = a_in[0*lda];
            SAXPY( a[1], bs[1], c );	a[1] = a_in[1*lda];
            SAXPY( a[2], bs[2], c );	a[2] = a_in[2*lda];
            SAXPY( a[3], bs[3], c );	a[3] = a_in[3*lda];

            a_in += 4*lda;
            SAXPY( a[0], bs[4], c );	a[0] = a_in[0*lda];
            SAXPY( a[1], bs[5], c );	a[1] = a_in[1*lda];
            SAXPY( a[2], bs[6], c );	a[2] = a_in[2*lda];
            SAXPY( a[3], bs[7], c );    a[3] = a_in[3*lda];

            a_in += 4*lda;
            SAXPY( a[0], bs[8], c );	a[0] = a_in[0*lda];
            SAXPY( a[1], bs[9], c );	a[1] = a_in[1*lda];
            SAXPY( a[2], bs[10], c );	a[2] = a_in[2*lda];
            SAXPY( a[3], bs[11], c );	a[3] = a_in[3*lda];

            a_in += 4*lda;
            SAXPY( a[0], bs[12], c );
            SAXPY( a[1], bs[13], c );
            SAXPY( a[2], bs[14], c );
            SAXPY( a[3], bs[15], c );

            b_in += 16;
            //counter += 16;
            device_hopper::batch_barrier();
        }
        //} while( counter < k );

        for(int i = 0; i < 16; i++){
            c_out[0] = alpha*c[i] + beta*c_out[0];
            c_out += ldc;
        }
    });
    // Register buffers and describe accesses
    pf.add_buffer_access_patterns(
        device_hopper::buf(a_in, direction::IN, pattern::ALL_OR_ANY),
        device_hopper::buf(b_in, direction::IN, pattern::ALL_OR_ANY),
        device_hopper::buf(c_out, direction::OUT, pattern::ALL_OR_ANY));
    // Add scalar kernel parameters
    pf.add_scalar_parameters(lda, ldb, ldc, k, alpha, beta);
    // Set optional tuning parameters and call run()
    pf.opt_set_batch_size(16, 4).opt_set_simple_indices(true).opt_set_is_idempotent(true).run();

    // Compute CRC
    boost::crc_32_type checksum;
    checksum.process_bytes(c_out, n * n * sizeof(float));
    std::cout << "Checksum: " << checksum.checksum() << std::endl;

    // Free buffers
    free(a_in);
    free(b_in);
    free(c_out);

    return EXIT_SUCCESS;
}