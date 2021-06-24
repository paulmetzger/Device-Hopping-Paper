{
    auto start_index_l = %START_INDEX_LAMBDA%;
    auto final_index_l = %FINAL_INDEX_LAMBDA%;
    const size_t start_index = start_index_l(0, BLOCK_SIZE_X);
    const size_t final_index = final_index_l(slice_size - 1, BLOCK_SIZE_X);
    const size_t size_in_bytes = ((final_index - start_index) + 1) * %ELEMENT_SIZE%;

    if (cudaMalloc(&h.cuda.%BUFFER_NAME%_d, size_in_bytes) != cudaSuccess)
        utils::exit_with_err("Could not allocate device buffers");
}