// Transfer results from the device to the host
if (dev == Device::GPU) {
    impl::transfer_data_from_device(
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