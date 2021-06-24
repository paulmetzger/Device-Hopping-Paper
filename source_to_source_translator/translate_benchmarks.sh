#!/bin/sh

CHECK() {
  echo -n "$1 $2 $3 $4 $5\n"
  $1 $2 $3 $4 $5
  ret_val=$?
  if [ $ret_val -eq 1 ]; then
    echo "The translation failed"
    exit $ret_val
  fi
}

mkdir -p ../benchmarks/opencl_files

CHECK ./translator.py --input ../benchmarks/rodinia_btree_find_k.cpp     --output ../benchmarks/generated_rodinia_btree_find_k.cu
CHECK ./translator.py --input ../benchmarks/rodinia_btree_find_range.cpp --output ../benchmarks/generated_rodinia_btree_find_range.cu
CHECK ./translator.py --input ../benchmarks/rodinia_nn.cpp               --output ../benchmarks/generated_rodinia_nn.cu
CHECK ./translator.py --input ../benchmarks/shoc_md5hash.cpp             --output ../benchmarks/generated_shoc_md5hash.cu
CHECK ./translator.py --input ../benchmarks/shoc_fft.cpp                 --output ../benchmarks/generated_shoc_fft.cu       --cl-compiler-flags="-cl-mad-enable"
CHECK ./translator.py --input ../benchmarks/shoc_gemm_nn.cpp             --output ../benchmarks/generated_shoc_gemm_nn.cu   --cl-compiler-flags="-cl-mad-enable"
CHECK ./translator.py --input ../benchmarks/shoc_ifft.cpp                --output ../benchmarks/generated_shoc_ifft.cu      --cl-compiler-flags="-cl-mad-enable"
CHECK ./translator.py --input ../benchmarks/shoc_reduction.cpp           --output ../benchmarks/generated_shoc_reduction.cu
CHECK ./translator.py --input ../benchmarks/shoc_spmv_scalar.cpp         --output ../benchmarks/generated_shoc_spmv.cu      --cl-compiler-flags="-cl-mad-enable"
CHECK ./translator.py --input ../benchmarks/example_matrix_vector_multiplication.cpp --output ../benchmarks/generated_example_matrix_vector_multiplication.cu
