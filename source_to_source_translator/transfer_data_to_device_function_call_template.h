// Transfer data to the device if the computation did not previously run on the device
// or transfer data if data is transferred on demand.
if (dev == Device::GPU) {
    impl::transfer_data_to_device(
            //%ABORT_SLICE_FLAG%
            //%BUFFERS%
            //%BUFFER_ELEMENT_COUNTS%
            //%BUFFER_ELEMENT_SIZES%
            current_offsets,
            current_slice_sizes,
            h,
            ctx);
    //%RETURN_IF_THE_SLICE_IS_ABORTED%
}