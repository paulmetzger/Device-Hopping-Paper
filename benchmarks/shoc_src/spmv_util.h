// Taken from the SHOC benchmark suite

#ifndef SPMV_UTIL_H_
#define SPMV_UTIL_H_

#include <cassert>
#include <iostream>
#include <fstream>

// Constants

// threshold for error in GPU results
static const double MAX_RELATIVE_ERROR = .02;

// alignment factor in terms of number of floats, used to enforce
// memory coalescing
static const int PAD_FACTOR = 16;

// size of atts buffer
static const int TEMP_BUFFER_SIZE = 1024;

// length of array for reading fields of mtx header
static const int FIELD_LENGTH = 128;

// If using a matrix market pattern, assign values from 0-MAX_RANDOM_VAL
static const float MAX_RANDOM_VAL = 10.0f;

struct Coordinate {
    int x;
    int y;
    float val;
};

inline int intcmp(const void *v1, const void *v2);
inline int coordcmp(const void *v1, const void *v2);
template <typename floatType>
void readMatrix(char *filename, floatType **val_ptr, int **cols_ptr,
                int **rowDelimiters_ptr, int *n, int *size);
template <typename floatType>
void printSparse(floatType *A, int n, int dim, int *cols, int *rowDelimiters);
template <typename floatType>
void convertToColMajor(floatType *A, int *cols, int dim, int *rowDelimiters,
                       floatType *newA, int *newcols, int *rl, int maxrl,
                       bool padded);
template <typename floatType>
void convertToPadded(floatType *A, int *cols, int dim, int *rowDelimiters,
                     floatType **newA_ptr, int **newcols_ptr, int *newIndices,
                     int *newSize);

// ****************************************************************************
// Function: fill
//
// Purpose:
//   Simple routine to initialize input array
//
// Arguments:
//   A: pointer to the array to initialize
//   n: number of elements in the array
//   maxi: specifies range of random values
//
// Programmer: Lukasz Wesolowski
// Creation: June 21, 2010
// Returns:  nothing
//
// ****************************************************************************
template <typename floatType>
void fill(floatType *A, const long n, const float maxi)
{
    for (int j = 0; j < n; j++)
    {
        A[j] = ((floatType) maxi * (rand() / (RAND_MAX + 1.0f)));
    }
}

// ****************************************************************************
// Function initRandomMatrix
//
// Purpose:
//   Assigns random positions to a given number of elements in a square
//   matrix, A.  The function encodes these positions in compressed sparse
//   row format.
//
// Arguments:
//   cols:          array for column indexes of elements (size should be = n)
//   rowDelimiters: array of size dim+1 holding indices to rows of A;
//                  last element is the index one past the last element of A
//   n:             number of nonzero elements in A
//   dim:           number of rows/columns in A
//
// Programmer: Kyle Spafford
// Creation: July 28, 2010
// Returns: nothing
//
// ****************************************************************************
void initRandomMatrix(
        long *cols,
        long *rowDelimiters,
        const long n,
        const long dim) {
    long nnzAssigned = 0;

    // Figure out the probability that a nonzero should be assigned to a given
    // spot in the matrix
    double prob = (double)n / ((double)dim * (double)dim);

    // Seed random number generator
    srand48(8675309L);

    // Randomly decide whether entry i,j gets a value, but ensure n values
    // are assigned
    bool fillRemaining = false;
    for (long i = 0; i < dim; i++)
    {
        rowDelimiters[i] = nnzAssigned;
        for (long j = 0; j < dim; j++)
        {
            long numEntriesLeft = (dim * dim) - ((i * dim) + j);
            long needToAssign   = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            if ((nnzAssigned < n && drand48() <= prob) || fillRemaining)
            {
                // Assign (i,j) a value
                cols[nnzAssigned] = j;
                nnzAssigned++;
            }
        }
    }
    // Observe the convention to put the number of non zeroes at the end of the
    // row delimiters array
    rowDelimiters[dim] = n;
    assert(nnzAssigned == n);
}

// ****************************************************************************
// Function printSparse
//
// Purpose:
//   Prints a sparse matrix in dense form for debugging purposes
//
// Arguments:
//   A: array holding the non-zero values for the matrix
//   n: number of elements in A
//   dim: number of rows/columns in the matrix
//   rowDelimiters: array of size dim+1 holding indices to rows of A;
//               last element is the index one past the last element of A
//
// Programmer: Lukasz Wesolowski
// Creation: June 22, 2010
// Returns: nothing
//
// ****************************************************************************
template <typename floatType>
void printSparse(floatType *A, int n, int dim, int *cols, int *rowDelimiters)
{

    int colIndex;
    int zero = 0;

    for (int i=0; i<dim; i++)
    {
        colIndex = 0;
        for (int j=rowDelimiters[i]; j<rowDelimiters[i+1]; j++)
        {
            while (colIndex++ < cols[j])
            {
                printf("%7d ", zero);
            }
            printf("%1.1e ", A[j]);;
        }
        while (colIndex++ < dim)
        {
            printf("%7d ", zero);
        }
        printf("\n");
    }

}

// ****************************************************************************
// Function: convertToColMajor
//
// Purpose:
//   Converts a sparse matrix representation whose data structures are
//   in row-major format into column-major format.
//
// Arguments:
//   A: array holding the non-zero values for the matrix in
//      row-major format
//   cols: array of column indices of the sparse matrix in
//         row-major format
//   dim: number of rows/columns in the matrix
//   rowDelimiters: array holding indices in A to rows of the sparse matrix
//   newA: input - buffer of size dim * maxrl
//         output - A in ELLPACK-R format
//   newcols: input - buffer of same size as newA
//            output - cols in ELLPACK-R format
//   rl: array storing length of every row of A
//   maxrl: maximum number of non-zero elements per row in A
//   padded: indicates whether newA should be padded so that the
//           number of rows divides PAD_FACTOR
//
// Programmer: Lukasz Wesolowski
// Creation: June 29, 2010
// Returns:
//   nothing directly
//   newA and newcols indirectly through pointers
// ****************************************************************************
template <typename floatType>
void convertToColMajor(floatType *A, int *cols, int dim, int *rowDelimiters,
                       floatType *newA, int *newcols, int *rl, int maxrl,
                       bool padded)
{
    int pad = 0;
    if (padded && dim % PAD_FACTOR != 0)
    {
        pad = PAD_FACTOR - dim % PAD_FACTOR;
    }

    int newIndex = 0;
    for (int j=0; j<maxrl; j++)
    {
        for (int i=0; i<dim; i++)
        {
            if (rowDelimiters[i] + j < rowDelimiters[i+1])
            {
                newA[newIndex] = A[rowDelimiters[i]+j];
                newcols[newIndex] = cols[rowDelimiters[i]+j];
            }
            else
            {
                newA[newIndex] = 0;
            }
            newIndex++;
        }
        if (padded)
        {
            for (int p=0; p<pad; p++)
            {
                newA[newIndex] = 0;
                newIndex++;
            }
        }
    }
}

// comparison functions used for qsort

inline int intcmp(const void *v1, const void *v2)
{
    return (*(int *)v1 - *(int *)v2);
}


inline int coordcmp(const void *v1, const void *v2)
{
    struct Coordinate *c1 = (struct Coordinate *) v1;
    struct Coordinate *c2 = (struct Coordinate *) v2;

    if (c1->x != c2->x)
    {
        return (c1->x - c2->x);
    }
    else
    {
        return (c1->y - c2->y);
    }
}

#endif // SPMV_UTIL_H_
