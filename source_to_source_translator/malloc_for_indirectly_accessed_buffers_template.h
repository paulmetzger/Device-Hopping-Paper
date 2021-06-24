{
    auto start_index_lambda = %START_INDEX_LAMBDA%;
    auto final_index_lambda = %FINAL_INDEX_LAMBDA%;
    const size_t start_thread_id = current_offsets[0] * BLOCK_SIZE_X;
    const size_t final_thread_id = start_thread_id + current_slice_sizes[0] * BLOCK_SIZE_X;
    const size_t start_index = start_index_lambda(start_thread_id);
    const size_t stop_index  = final_index_lambda(final_thread_id);
    const size_t indirect_start_index = %INTERMEDIATE_BUFFER%[start_index];
    const size_t indirect_stop_index  = %INTERMEDIATE_BUFFER%[stop_index];
    const size_t elements = indirect_stop_index - indirect_start_index;
    const size_t required_buffer_size = elements * %ELEMENT_SIZE%;
    if (%BUFFER_NAME%_current_size_in_bytes < required_buffer_size) {
        cudaError err = cudaFree(h.cuda.%BUFFER_NAME%_d);
        if (err != cudaSuccess) utils::exit_with_err("Could not reallocate buffer '%BUFFER_NAME%'");
        err = cudaMalloc(&h.cuda.%BUFFER_NAME%_d, required_buffer_size);
        if (err != cudaSuccess) utils::exit_with_err("Could not reallocate buffer '%BUFFER_NAME%'");
        %BUFFER_NAME%_current_size_in_bytes = required_buffer_size;
    }
    %DATA_TRANSFER%
}