{
    auto start_index_l = %START_INDEX_LAMBDA%;
    auto final_index_l = %FINAL_INDEX_LAMBDA%;
    const size_t start_index       = start_index_l(current_offsets[0], BLOCK_SIZE_X);
    const size_t final_index       = final_index_l(current_offsets[0] + current_slice_sizes[0] - 1, BLOCK_SIZE_X);
    const size_t bytes_to_transfer = (final_index - start_index + 1) * %ELEMENT_SIZE%;
    char* h_ptr = (char*) (%BUFFER_NAME_HOST% + start_index);
    char* d_ptr = (char*) (h.cuda.%BUFFER_NAME_DEVICE% + %DEVICE_BUFFER_INDEX%);

    // Copy results from the device to the host memory
    if (cudaMemcpy(%DESTINATION_POINTER%,
                   %SOURCE_POINTER%,
                   bytes_to_transfer,
                   %TRANSFER_DIRECTION%) != cudaSuccess)
        utils::exit_with_err("Could not copy '%BUFFER_NAME_HOST%' (%TRANSFER_DIRECTION%)");
}