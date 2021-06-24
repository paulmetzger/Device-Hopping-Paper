/**
 * Some part of the source code is from the SHOC benchmark suite.
 *
 * Original authors:
 * Jeremy Meredith
 */

#include <boost/math/special_functions/next.hpp>
#include <boost/program_options.hpp>
#include <cstdlib>
#include <iostream>
#include <tuple>

#include <omp.h>
#include <sys/mman.h>

#include "../device_hopper/core.h"

using namespace device_hopper;

#define REPOSITORY_PATH std::string(std::getenv("PLASTICITY_ROOT"))
#define PREFERRED_DEVICE GPU

#define LEFTROTATE(x, c) (((x) << (c)) | ((x) >> (32 - (c))))
#define F(x, y, z) ((x & y) | ((~x) & z))
#define G(x, y, z) ((x & z) | ((~z) & y))
#define H(x, y, z) (x ^ y ^ z)
#define I(x, y, z) (y ^ (x | (~z)))

// This version ignores the mapping of a/b/c/d to v/x/y/z and simply
// uses a temporary variable to keep the interpretation of a/b/c/d
// consistent.  Whether this one or the previous one performs better
// probably depends on the compiler....
#define ROUND_USING_TEMP_VARS(w, r, k, v, x, y, z, func)         \
{                                                                \
    a = a + func(b,c,d) + k + w;                                 \
    unsigned int temp = d;                                       \
    d = c;                                                       \
    c = b;                                                       \
    b = b + LEFTROTATE(a, r);                                    \
    a = temp;                                                    \
}

// Here, we pick which style of ROUND we use.
#define ROUND ROUND_USING_TEMP_VARS

void IndexToKey(unsigned int index,
                int byteLength,
                int valsPerByte,
                unsigned char vals[8]) {
    // loop pointlessly unrolled to avoid CUDA compiler complaints
    // about unaligned accesses (!?) on older compute capabilities
    vals[0] = index % valsPerByte;
    index /= valsPerByte;

    vals[1] = index % valsPerByte;
    index /= valsPerByte;

    vals[2] = index % valsPerByte;
    index /= valsPerByte;

    vals[3] = index % valsPerByte;
    index /= valsPerByte;

    vals[4] = index % valsPerByte;
    index /= valsPerByte;

    vals[5] = index % valsPerByte;
    index /= valsPerByte;

    vals[6] = index % valsPerByte;
    index /= valsPerByte;

    vals[7] = index % valsPerByte;
    //index /= valsPerByte;
}

int FindKeyspaceSize(int byteLength, int valsPerByte) {
    int keyspace = 1;
    for (int i = 0; i < byteLength; ++i) {
        if (keyspace >= 0x7fffffff / valsPerByte) {
            // error, we're about to overflow a signed int
            return -1;
        }
        keyspace *= valsPerByte;
    }
    return keyspace;
}

inline void md5_2words(unsigned int *words,
                       unsigned int len,
                       unsigned int *digest) {
    // For any block but the first one, these should be passed in, not
    // initialized, but we are assuming we only operate on a single block.
    unsigned int h0 = 0x67452301;
    unsigned int h1 = 0xefcdab89;
    unsigned int h2 = 0x98badcfe;
    unsigned int h3 = 0x10325476;

    unsigned int a = h0;
    unsigned int b = h1;
    unsigned int c = h2;
    unsigned int d = h3;

    unsigned int WL = len * 8;
    unsigned int W0 = words[0];
    unsigned int W1 = words[1];

    switch (len) {
        case 0:
            W0 |= 0x00000080;
            break;
        case 1:
            W0 |= 0x00008000;
            break;
        case 2:
            W0 |= 0x00800000;
            break;
        case 3:
            W0 |= 0x80000000;
            break;
        case 4:
            W1 |= 0x00000080;
            break;
        case 5:
            W1 |= 0x00008000;
            break;
        case 6:
            W1 |= 0x00800000;
            break;
        case 7:
            W1 |= 0x80000000;
            break;
    }

    // args: word data, per-round shift amt, constant, 4 vars, function macro
    ROUND(W0, 7, 0xd76aa478, a, b, c, d, F);
    ROUND(W1, 12, 0xe8c7b756, d, a, b, c, F);
    ROUND(0, 17, 0x242070db, c, d, a, b, F);
    ROUND(0, 22, 0xc1bdceee, b, c, d, a, F);
    ROUND(0, 7, 0xf57c0faf, a, b, c, d, F);
    ROUND(0, 12, 0x4787c62a, d, a, b, c, F);
    ROUND(0, 17, 0xa8304613, c, d, a, b, F);
    ROUND(0, 22, 0xfd469501, b, c, d, a, F);
    ROUND(0, 7, 0x698098d8, a, b, c, d, F);
    ROUND(0, 12, 0x8b44f7af, d, a, b, c, F);
    ROUND(0, 17, 0xffff5bb1, c, d, a, b, F);
    ROUND(0, 22, 0x895cd7be, b, c, d, a, F);
    ROUND(0, 7, 0x6b901122, a, b, c, d, F);
    ROUND(0, 12, 0xfd987193, d, a, b, c, F);
    ROUND(WL, 17, 0xa679438e, c, d, a, b, F);
    ROUND(0, 22, 0x49b40821, b, c, d, a, F);

    ROUND(W1, 5, 0xf61e2562, a, b, c, d, G);
    ROUND(0, 9, 0xc040b340, d, a, b, c, G);
    ROUND(0, 14, 0x265e5a51, c, d, a, b, G);
    ROUND(W0, 20, 0xe9b6c7aa, b, c, d, a, G);
    ROUND(0, 5, 0xd62f105d, a, b, c, d, G);
    ROUND(0, 9, 0x02441453, d, a, b, c, G);
    ROUND(0, 14, 0xd8a1e681, c, d, a, b, G);
    ROUND(0, 20, 0xe7d3fbc8, b, c, d, a, G);
    ROUND(0, 5, 0x21e1cde6, a, b, c, d, G);
    ROUND(WL, 9, 0xc33707d6, d, a, b, c, G);
    ROUND(0, 14, 0xf4d50d87, c, d, a, b, G);
    ROUND(0, 20, 0x455a14ed, b, c, d, a, G);
    ROUND(0, 5, 0xa9e3e905, a, b, c, d, G);
    ROUND(0, 9, 0xfcefa3f8, d, a, b, c, G);
    ROUND(0, 14, 0x676f02d9, c, d, a, b, G);
    ROUND(0, 20, 0x8d2a4c8a, b, c, d, a, G);

    ROUND(0, 4, 0xfffa3942, a, b, c, d, H);
    ROUND(0, 11, 0x8771f681, d, a, b, c, H);
    ROUND(0, 16, 0x6d9d6122, c, d, a, b, H);
    ROUND(WL, 23, 0xfde5380c, b, c, d, a, H);
    ROUND(W1, 4, 0xa4beea44, a, b, c, d, H);
    ROUND(0, 11, 0x4bdecfa9, d, a, b, c, H);
    ROUND(0, 16, 0xf6bb4b60, c, d, a, b, H);
    ROUND(0, 23, 0xbebfbc70, b, c, d, a, H);
    ROUND(0, 4, 0x289b7ec6, a, b, c, d, H);
    ROUND(W0, 11, 0xeaa127fa, d, a, b, c, H);
    ROUND(0, 16, 0xd4ef3085, c, d, a, b, H);
    ROUND(0, 23, 0x04881d05, b, c, d, a, H);
    ROUND(0, 4, 0xd9d4d039, a, b, c, d, H);
    ROUND(0, 11, 0xe6db99e5, d, a, b, c, H);
    ROUND(0, 16, 0x1fa27cf8, c, d, a, b, H);
    ROUND(0, 23, 0xc4ac5665, b, c, d, a, H);

    ROUND(W0, 6, 0xf4292244, a, b, c, d, I);
    ROUND(0, 10, 0x432aff97, d, a, b, c, I);
    ROUND(WL, 15, 0xab9423a7, c, d, a, b, I);
    ROUND(0, 21, 0xfc93a039, b, c, d, a, I);
    ROUND(0, 6, 0x655b59c3, a, b, c, d, I);
    ROUND(0, 10, 0x8f0ccc92, d, a, b, c, I);
    ROUND(0, 15, 0xffeff47d, c, d, a, b, I);
    ROUND(W1, 21, 0x85845dd1, b, c, d, a, I);
    ROUND(0, 6, 0x6fa87e4f, a, b, c, d, I);
    ROUND(0, 10, 0xfe2ce6e0, d, a, b, c, I);
    ROUND(0, 15, 0xa3014314, c, d, a, b, I);
    ROUND(0, 21, 0x4e0811a1, b, c, d, a, I);
    ROUND(0, 6, 0xf7537e82, a, b, c, d, I);
    ROUND(0, 10, 0xbd3af235, d, a, b, c, I);
    ROUND(0, 15, 0x2ad7d2bb, c, d, a, b, I);
    ROUND(0, 21, 0xeb86d391, b, c, d, a, I);

    h0 += a;
    h1 += b;
    h2 += c;
    h3 += d;

    // write the final result out
    digest[0] = h0;
    digest[1] = h1;
    digest[2] = h2;
    digest[3] = h3;
}

DEVICE_HOPPER_MAIN(int argc, char *argv[]) {
    DEVICE_HOPPER_SETUP

    // Parse CLI parameters
    boost::program_options::options_description desc("Options");
    desc.add_options()("problem-size", boost::program_options::value<long>(), "Sample count");
    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    size_t problem_size = 0;
    if (vm.count("problem-size") == 0) {
        std::cerr << "Error: Problem size is missing." << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        // This parameter is passed to the application by the Python scripts that automate the experiments.
        problem_size = vm["problem-size"].as<long>();
    }

#if defined(SOURCE_TO_SOURCE_TRANSLATED)
    // This sets the number of blocks in a slice as slice size as expected by the parallel for.
    // The hand implementation uses the number of iteration points in a slice as slice size.
    // To keep the database tables and configs consistent we continue to use the old slice sizes in terms of iteration points
    // in the configs and databases but convert them then here in the source code.
    context.sched.set_cpu_slice_sizes(context.sched.get_cpu_x_slice_size() / 256, 1);
    context.sched.set_gpu_slice_sizes(context.sched.get_gpu_x_slice_size() / 256, 1);
#endif

    int byte_length = 0;
    int vals_per_byte = 0;
    if (problem_size == 24010240) {
        byte_length = 5;
        vals_per_byte = 70;
    } else if (problem_size == 9765632) {
        byte_length = 6;
        vals_per_byte = 25;
    } else if (problem_size == 5153792) {
        byte_length = 6;
        vals_per_byte = 22;
    } else if (problem_size == 1500672) {
        byte_length = 5;
        vals_per_byte = 35;
    } else if (problem_size == 1000192) {
        byte_length = 7;
        vals_per_byte = 10;
    } else {
        std::cerr << "Error this problem size is not supported" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Compute kernel parameters and the key and index that we are looking for.
    // The key and index that we want to compute is stored in "random_key" and "random_index".
    const int keyspace = FindKeyspaceSize(byte_length, vals_per_byte);
    srandom(1234);
    int random_index = rand() % keyspace;
    unsigned char random_key[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    unsigned int random_digest[4];
    IndexToKey(random_index, byte_length, vals_per_byte, random_key);
    md5_2words((unsigned int *) random_key, byte_length, random_digest);

    // These arrays and variables will hold the results
    size_t found_digest_size   = 4;
    unsigned int *found_digest = (unsigned int*) device_hopper::malloc(found_digest_size, sizeof(unsigned int));
    for (int i = 0; i < found_digest_size; ++i) found_digest[i] = 0;
    size_t found_index_size    = 1;
    int *found_index           = (int*) device_hopper::malloc(found_index_size, sizeof(int));
    *found_index               = 0;
    size_t foound_key_size     = 8;
    unsigned char *found_key   = (unsigned char*) device_hopper::malloc(foound_key_size, sizeof(unsigned char));
    for (int i = 0; i < foound_key_size; ++i) found_key[i] = 0;
    unsigned int random_digest_0 = random_digest[0];
    unsigned int random_digest_1 = random_digest[1];
    unsigned int random_digest_2 = random_digest[2];
    unsigned int random_digest_3 = random_digest[3];

    parallel_for pf(0, problem_size, [&]DEVICE_HOPPER_LAMBDA() {
        int threadid = GET_ITERATION();

        int startindex = threadid * vals_per_byte;
        unsigned char key[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        IndexToKey(startindex, byte_length, vals_per_byte, key);

        unsigned int digest[4];
        int j;
        for (j = 0; j < vals_per_byte && startindex + j < keyspace; ++j) {
            //unsigned int digest[4];
            md5_2words((unsigned int *) key, byte_length, digest);
            if (digest[0] == random_digest_0 &&
                digest[1] == random_digest_1 &&
                digest[2] == random_digest_2 &&
                digest[3] == random_digest_3) {
                *found_index = startindex + j;
                found_key[0] = key[0];
                found_key[1] = key[1];
                found_key[2] = key[2];
                found_key[3] = key[3];
                found_key[4] = key[4];
                found_key[5] = key[5];
                found_key[6] = key[6];
                found_key[7] = key[7];
                found_digest[0] = digest[0];
                found_digest[1] = digest[1];
                found_digest[2] = digest[2];
                found_digest[3] = digest[3];
            }
            ++key[0];
        }
    });
    // Register buffers and specify access patterns
    pf.add_buffer_access_patterns(
        device_hopper::buf(found_digest, direction::OUT, pattern::ALL_OR_ANY),
        device_hopper::buf(found_key,    direction::OUT, pattern::ALL_OR_ANY),
        device_hopper::buf(found_index,  direction::OUT, pattern::ALL_OR_ANY));
    // Add scalar kernel parameters
    pf.add_scalar_parameters(random_digest_0, random_digest_1, random_digest_2, random_digest_3, keyspace, byte_length, vals_per_byte);
    // Set optional tuning parameters and call run()
    pf.opt_set_batch_size(256).opt_set_is_idempotent(true).run();

    if (*found_index < 0) {
        std::cout << "Error: Could not find an index" << std::endl;
        std::cout << "Error: The results are incorrect" << std::endl;
        return EXIT_FAILURE;
    } else if (*found_index != random_index) {
        std::cout << "Error: The computed index is not correct" << std::endl;
        std::cout << "Computed index: " << *found_index << std::endl;
        std::cout << "Expected index: " << random_index << std::endl;
        std::cout << "Error: The results are incorrect" << std::endl;
        return EXIT_FAILURE;
    } else if (found_key[0] != random_key[0] ||
               found_key[1] != random_key[1] ||
               found_key[2] != random_key[2] ||
               found_key[3] != random_key[3] ||
               found_key[4] != random_key[4] ||
               found_key[5] != random_key[5] ||
               found_key[6] != random_key[6] ||
               found_key[7] != random_key[7]) {
        std::cout << "Error: The computed key is not correct" << std::endl;
        std::cout << "Computed key, expected key " << std::endl;
        for (size_t i = 0; i < 8; ++i) std::cout << found_key[i] << ", " << random_key[i] << std::endl;
        std::cout << std::endl;
        std::cout << "Error: The results are incorrect" << std::endl;
        return EXIT_FAILURE;
    } else if (found_digest[0] != random_digest[0] ||
               found_digest[1] != random_digest[1] ||
               found_digest[2] != random_digest[2] ||
               found_digest[3] != random_digest[3]) {
        std::cout << "Error: The computed digest is not correct" << std::endl;
        std::cout << "Computed digest, expected digest " << std::endl;
        for (size_t i = 0; i < 4; ++i) std::cout << found_digest[i] << ", " << random_digest[i] << std::endl;
        std::cout << std::endl;
        std::cout << "Error: The results are incorrect" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Info: The results are correct" << std::endl;

    return EXIT_SUCCESS;
}