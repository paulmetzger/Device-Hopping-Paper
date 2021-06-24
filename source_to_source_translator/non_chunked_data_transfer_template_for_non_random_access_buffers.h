{
    char* h_ptr = (char*) (%BUFFER_NAME_HOST% + current_offsets[0] * %OFFSET_SHIFT% * \
                           %NUMBER_OF_ELEMENTS_ACCESSED_PER_BLOCK_OR_THREAD_ID%);
    const size_t _bytes_to_transfer = current_slice_sizes[0] * %OFFSET_SHIFT% * %ELEMENT_SIZE% *
                                     %NUMBER_OF_ELEMENTS_ACCESSED_PER_BLOCK_OR_THREAD_ID%;
    char* d_ptr                             = (char*) h.cuda.%BUFFER_NAME_DEVICE%;

    // Copy results from the device to the host memory
    if (cudaMemcpy(%DESTINATION_POINTER%,
                   %SOURCE_POINTER%,
                   _bytes_to_transfer,
                   %TRANSFER_DIRECTION%) != cudaSuccess)
        utils::exit_with_err("Could not copy '%BUFFER_NAME_HOST%' (%TRANSFER_DIRECTION%)");
}