{
    const size_t _bytes_to_transfer = %BUFFER_ELEMENT_COUNT% * %ELEMENT_SIZE%;

    // Copy results from the device to the host memory
    if (cudaMemcpy(%DESTINATION_BUFFER%,
                   %SOURCE_BUFFER%,
                   _bytes_to_transfer,
                   %TRANSFER_DIRECTION%) != cudaSuccess)
        utils::exit_with_err("Could not copy '%BUFFER_NAME_HOST%' (%TRANSFER_DIRECTION%)");
}