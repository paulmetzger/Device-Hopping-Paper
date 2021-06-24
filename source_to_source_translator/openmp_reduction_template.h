#pragma omp parallel for reduction(%REDUCTION_OPERATOR%:%REDUCTION_DESTINATION%)
for (size_t i = offsets[0]; i < offsets[0] + slice_sizes[0]; ++i)
    %REDUCTION_DESTINATION% %REDUCTION_OPERATOR%= %INPUT_BUFFER_NAME%[i];