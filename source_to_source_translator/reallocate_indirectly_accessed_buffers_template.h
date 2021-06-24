if (h.cuda.%BUFFER_NAME%_d_size < bytes_to_transfer) {
    cudaError err = cudaFree(h.cuda.%BUFFER_NAME%_d);
    if (err != cudaSuccess) utils::exit_with_err("Could not free h.cuda.%BUFFER_NAME%_d");
    err = cudaMalloc(&h.cuda.%BUFFER_NAME%_d, bytes_to_transfer);
    if (err != cudaSuccess) utils::exit_with_err("Could not allocate h.cuda.%BUFFER_NAME%_d");
    h.cuda.%BUFFER_NAME%_d_size = bytes_to_transfer;
}