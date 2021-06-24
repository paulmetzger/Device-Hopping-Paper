{
    char* h_ptr                             = (char*) %SOURCE_BUFFER%;
    const size_t bytes_to_transfer          = %BUFFER_ELEMENT_COUNT% * %ELEMENT_SIZE%;
    const size_t chunk_size_bytes           = 1024u * 1024u * ctx.data_transf_slice_size_mb;
    const size_t buf_chunks                 = bytes_to_transfer / chunk_size_bytes;
    const size_t chunk_size_in_buf_elements = chunk_size_bytes / %ELEMENT_SIZE%;
    const size_t buf_remaining_bytes        = bytes_to_transfer % chunk_size_bytes;
    char* d_ptr                             = (char*) %DESTINATION_BUFFER%;

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
    if (cudaMemcpy(%DESTINATION_POINTER%, %SOURCE_POINTER%, buf_remaining_bytes + (buf_chunks > 0 ? chunk_size_bytes : 0),
                   %TRANSFER_DIRECTION%) != cudaSuccess)
        utils::exit_with_err("Could not copy '%BUFFER_NAME_HOST%' (%TRANSFER_DIRECTION%)");
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
