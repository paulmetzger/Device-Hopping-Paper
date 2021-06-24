/**
     * This function transfers results from the GPU to the CPU.
     */
    inline void transfer_data_from_device(
            //%ABORT_SLICE_FLAG_DECL%
            //%BUFFER_PARAMETER_DECLS%
            //%BUFFER_ELEMENT_COUNTS_PARAMETER_DECLS%
            //%BUFFER_ELEMENT_SIZES_PARAMETER_DECLS%
            const size_t (&current_offsets)[2],
            const size_t (&current_slice_sizes)[2],
            handles& h,
            plasticity::setup::Context& ctx) {
#ifdef DEBUG
        std::cout << "DEBUG: Transferring data from device... " << std::flush;
#endif
        //%DATA_TRANSFER_FROM_DEVICE%
#ifdef DEBUG
        std::cout << "Done" << std::endl;
#endif
    }