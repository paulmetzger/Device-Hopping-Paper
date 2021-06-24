/**
 * This function transfers data to the GPU memory with clEnqueueWriteBufferRect.
 * Do not call this function when we execute on the CPU.
 * Call this function for each slice if the kernel uses slicing aware data transfers, and
 * call it only once if not.
 */
inline void transfer_data_to_device(
        //%ABORT_SLICE_FLAG_DECL%
        //%BUFFER_PARAMETER_DECLS%
        //%BUFFER_ELEMENT_COUNTS_PARAMETER_DECLS%
        //%BUFFER_ELEMENT_SIZES_PARAMETER_DECLS%
        const size_t (&current_offsets)[2],
        const size_t (&current_slice_sizes)[2],
        handles& h,
        plasticity::setup::Context& ctx) {
#ifdef DEBUG
    std::cout << "DEBUG: Transferring data to device... " << std::flush;
#endif
    //%DATA_TRANSFER_TO_DEVICE%
#ifdef DEBUG
    std::cout << "Done" << std::endl;
#endif
}