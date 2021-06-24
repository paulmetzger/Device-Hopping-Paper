from enum import Enum


class access_pattern(Enum):
    BY_THREAD_ID = 1
    BY_BLOCK_ID  = 2
    CONTINUOUS   = 3
    INDIRECT     = 4
    RANDOM       = 5
    REDUCTION    = 6
    REDUCTION_INTERMEDIATE_RESULTS = 7
    SUCCESSIVE_SUBSECTIONS = 8


class access_direction(Enum):
    IN    = 1
    OUT   = 2
    INOUT = 3


class address_space(Enum):
    GLOBAL = 1
    SHARED_WITHIN_GRAIN = 2


class data_kind(Enum):
    INTERIM_RESULTS = 1


class Buffer:
    def __init__(
            self,
            access_pattern: access_pattern,
            access_direction: access_direction,
            address_space: address_space,
            buffer_name: str,
            element_count: str,
            type: str):
        self.access_pattern   = access_pattern
        self.access_direction = access_direction
        self.address_space    = address_space
        self.buffer_name      = buffer_name
        self.contains_indices = False
        self.data_kind        = None
        self.element_count    = element_count
        self.intermediate_buffer_for_indirect_accesses = None
        self.is_accessed_indirectly = False
        self.lambda_to_compute_start_index_for_indirect_accesses = None
        self.lambda_to_compute_final_index_for_indirect_accesses = None
        self.lambda_to_compute_start_index_for_continuous_accesses = None
        self.lambda_to_compute_final_index_for_continuous_accesses = None
        self.number_of_elements_accessed_per_block_or_thread_id = 1
        self.number_of_overlapping_accesses = 0
        self.type               = type
        self.use_texture_on_gpu = False

    def opencl_address_space_qualifier(self):
        if self.address_space == address_space.GLOBAL:
            return '__global'
        elif self.address_space == address_space.SHARED_WITHIN_GRAIN:
            return '__local'
        else:
            print('Error: Unknown address space')
            exit(1)

    def set_data_kind(self, kind: data_kind):
        self.data_kind = kind

    def set_intermediate_buffer_for_indirect_accesses(self, intermediate_buffer):
        self.intermediate_buffer_for_indirect_accesses = intermediate_buffer

    def set_is_accessed_indirectly(self, is_accessed_indirectly):
        self.is_accessed_indirectly = is_accessed_indirectly

    def set_contains_indices(self, contains_indices):
        self.contains_indices = contains_indices

    def set_lambda_to_compute_start_index_for_indirect_accesses(self, start_lambda):
        self.lambda_to_compute_start_index_for_indirect_accesses = start_lambda

    def set_lambda_to_compute_final_index_for_indirect_accesses(self, final_lambda):
        self.lambda_to_compute_final_index_for_indirect_accesses = final_lambda

    def set_lambda_to_compute_start_index_for_continuous_accesses(self, start_lambda):
        self.lambda_to_compute_start_index_for_continuous_accesses = start_lambda

    def set_lambda_to_compute_final_index_for_continuous_accesses(self, final_lambda):
        self.lambda_to_compute_final_index_for_continuous_accesses = final_lambda

    def set_number_of_elements_accessed_per_block_or_thread_id(
            self, number_of_elements_accessed_per_block_or_thread_id: int):
        self.number_of_elements_accessed_per_block_or_thread_id = number_of_elements_accessed_per_block_or_thread_id

    def set_number_of_overlapping_memory_accesses(self, overlapping_accesses: int):
        self.number_of_overlapping_accesses = overlapping_accesses

    def set_use_texture_on_gpu(self, use_texture):
        self.use_texture_on_gpu = use_texture


