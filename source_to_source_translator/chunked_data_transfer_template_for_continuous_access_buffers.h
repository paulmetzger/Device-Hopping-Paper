{
    auto start_index_l = %START_INDEX_LAMBDA%;
    auto final_index_l = %FINAL_INDEX_LAMBDA%;
    const size_t start_index = start_index_l(current_offsets[0], BLOCK_SIZE_X);
    const size_t final_index = final_index_l(current_offsets[0] + current_slice_sizes[0] - 1, BLOCK_SIZE_X);
    const size_t bytes_to_transfer          = (final_index - start_index + 1) * %ELEMENT_SIZE%;
    //%CHECK_IF_INDIRECTLY_ACCESSED_BUFFERS_NEED_TO_BE_RESIZED%
    char* h_ptr                             = (char*) (%BUFFER_NAME_HOST% + start_index);
    const size_t chunk_size_bytes           = 1024u * 1024u * ctx.data_transf_slice_size_mb;
    const size_t buf_chunks                 = bytes_to_transfer / chunk_size_bytes;
    const size_t buf_remaining_bytes        = bytes_to_transfer % chunk_size_bytes;
    char* d_ptr                             = (char*) (h.cuda.%BUFFER_NAME_DEVICE% + %DEVICE_BUFFER_INDEX%);

#if defined(DEBUG)
    std::cout << "Chunks to transfer: " << buf_chunks << std::endl;
#endif
    cudaError_t err;
    if (buf_chunks != 0) {
        for (size_t chunk = 0; chunk < buf_chunks - 1; ++chunk) {
#ifdef DEBUG
            std::cout << "DEBUG " << utils::convert_device_to_str(h.device)
                      << ": Transferring data chunk to device... " << std::endl;
#endif

            if ((err = cudaMemcpy(%DESTINATION_POINTER%, %SOURCE_POINTER%, chunk_size_bytes, %TRANSFER_DIRECTION%)) != cudaSuccess) {
                std::cerr << cudaGetErrorString(err) << std::endl;
                utils::exit_with_err("Could not copy '%BUFFER_NAME_HOST%' (%TRANSFER_DIRECTION%)");
            }

            if (ctx.sched.kill_current_slice(h.device)) {
                transfer_random_access_buffers_from_device(
                    //%BUFFERS%
                    //%BUFFER_ELEMENT_COUNTS%
                    //%BUFFER_ELEMENT_SIZES%
                    h);
                cleanup(&h);
                ctx.sched.free_device(h.device);
                abort_slice = true;
                return;
            }

            h_ptr += chunk_size_bytes;
            d_ptr += chunk_size_bytes;
        }
    }

#ifdef DEBUG
    std::cout << "DEBUG " << utils::convert_device_to_str(h.device)
              << ": Transferring remaining data chunk to device... " << std::endl;
#endif
    err = cudaMemcpy(%DESTINATION_POINTER%, %SOURCE_POINTER%, buf_remaining_bytes + (buf_chunks > 0 ? chunk_size_bytes : 0),
                   %TRANSFER_DIRECTION%);
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err) << std::endl;
        utils::exit_with_err("Could not copy '%BUFFER_NAME_HOST%' (%TRANSFER_DIRECTION%)");
    }
    if (ctx.sched.kill_current_slice(h.device)) {
        transfer_random_access_buffers_from_device(
                //%BUFFERS%
                //%BUFFER_ELEMENT_COUNTS%
                //%BUFFER_ELEMENT_SIZES%
                h);
        cleanup(&h);
        ctx.sched.free_device(h.device);
        abort_slice = true;
        return;
    }
}
