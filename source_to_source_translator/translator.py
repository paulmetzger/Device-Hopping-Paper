#!/usr/bin/env python3

from typing import Dict, List, Set, Tuple

import argparse
import copy
import math
import os
import re
import subprocess
import sys
import traceback

from device import Device
import buffer_spec
import reduction

STANDARD_TYPES = ['char', 'size_t', 'int', 'double', 'float', 'long', 'unsigned int']

cuda_memcpy_template = 'if ((err = cudaMemcpy(%DESTINATION_BUFFER%, %SOURCE_BUFFER%, %ELEMENT_SIZE% * %ELEMENT_COUNT% , %TRANSFER_DIRECTION%)) != cudaSuccess) {\n'\
                              'std::cerr << cudaGetErrorString(err) << std::endl;\n'\
                              'utils::exit_with_err("Could not copy \'%SOURCE_BUFFER%\' to \'%DESTINATION_BUFFER%\' (cudaMemcpyHostToDevice)");\n'\
                           '}'


def scalar_parameter_is_buffer_size(buffers: List[buffer_spec.Buffer], scalar_parameter: str) -> bool:
    for b in buffers:
        if b.element_count == scalar_parameter:
            return True

    return False


def extract_if_batch_size_is_set(code: str, is_reduction: bool) -> bool:
    return ('opt_set_batch_size' in code) or is_reduction


def extract_whether_the_kernel_has_2d_iteration_space(code: str):
    parallel_for_instantiation = 'parallel_for (\w|\d)*\(.*\)'
    for line in code.split('\n'):
        match = re.search(parallel_for_instantiation, line)
        if match:
            parallel_for_constructor_call = line[match.start(): match.end()]
            if 'make_tuple(' in parallel_for_constructor_call:
                return True
    return False


def extract_if_indices_are_used_only_for_memory_accesses(code: str) -> bool:
    return re.search('opt_set_simple_indices', code) != None


def extract_kernel_code(code: str) -> str:
    kernel_code_regex = r'(?<=DEVICE\_HOPPER\_LAMBDA\(\)\s\{)(\n.*)*(?=\}\);)'
    match = re.search(kernel_code_regex, code)
    if match == None:
        kernel_code_regex = r'(?<=DEVICE\_HOPPER\_LAMBDA\(\) mutable \{)(\n.*)*(?=\}\);)'
        match = re.search(kernel_code_regex, code)
    kernel_code = code[match.start():match.end()]
    return kernel_code
    '''s = code.find(lambda_start) + len(lambda_start)
    e = s
    open_curly_braces = 1
    while not code[e] == ')' or not open_curly_braces == 0:
        if code[e] == '{':
            open_curly_braces += 1
        elif code[e] == '}':
            open_curly_braces -= 1
        e += 1
    return code[s:e - 1].strip()'''


def extract_batch_size(code: str, is_2d_kernel: bool, is_reduction: bool):
    if is_reduction:
        return tuple([256, 1])
    for line in code.split('\n'):
        if 'opt_set_batch_size' in line:
            s = line.find('opt_set_batch_size(')
            s += len('opt_set_batch_size(')
            e = s
            while line[e] != ')':
                e += 1
            if is_2d_kernel:
                return tuple([int(size.strip()) for size in line[s:e].split(',')])
            else:
                return tuple([int(line[s:e]), 1])
    print('Error: No block size set')
    exit(1)


def access_pattern_str_to_enum(access_pattern: str) -> buffer_spec.access_pattern:
    all_or_any            = 'pattern::ALL_OR_ANY'
    by_thread_id          = 'pattern::BY_THREAD_ID'
    by_block_id           = 'pattern::BY_BLOCK_ID'
    continuous            = 'pattern::BATCH_CONTINUOUS'
    indirect              = 'pattern::INDIRECT'
    random                = 'pattern::RANDOM'
    reduction             = 'pattern::REDUCTION'
    successive_subsection = 'pattern::SUCCESSIVE_SUBSECTIONS'
    access_pattern = access_pattern.strip()
    if all_or_any              == access_pattern: return buffer_spec.access_pattern.RANDOM
    elif by_block_id           == access_pattern: return buffer_spec.access_pattern.BY_BLOCK_ID
    elif by_thread_id          == access_pattern: return buffer_spec.access_pattern.BY_THREAD_ID
    elif continuous            in access_pattern: return buffer_spec.access_pattern.CONTINUOUS
    elif indirect              == access_pattern: return buffer_spec.access_pattern.INDIRECT
    elif random                == access_pattern: return buffer_spec.access_pattern.RANDOM
    elif reduction             == access_pattern: return buffer_spec.access_pattern.REDUCTION
    elif successive_subsection in access_pattern: return buffer_spec.access_pattern.SUCCESSIVE_SUBSECTIONS
    else:
        print('Error: Unknown access pattern')
        print(access_pattern)
        traceback.print_stack()
        sys.exit(1)


def access_direction_str_to_enum(access_direction: str) -> buffer_spec.access_direction:
    in_dir    = 'direction::IN'
    out_dir   = 'direction::OUT'
    inout_dir = 'direction::IN_OUT'
    access_direction = access_direction.strip()
    if access_direction == in_dir: return buffer_spec.access_direction.IN
    elif access_direction == out_dir: return buffer_spec.access_direction.OUT
    elif access_direction == inout_dir: return buffer_spec.access_direction.INOUT
    else:
        print('Error: Unknown access direction')
        sys.exit(1)


def extract_number_of_elements_accessed_per_batch(line: str) -> int:
    r = re.search('(?<=pattern::SUCCESSIVE_SUBSECTIONS\()(\d| |\*|\+|\\|\-|\(|\))*|(?=\))', line)
    if r:
        return eval(r.group()[:-1])
    else:
        return 1


def extract_number_of_elements_accessed_per_thread_id(line: str) -> int:
    r = re.search('(?<=pattern::ELEMENTS_ACCESSED_PER_THREAD_ID\()\d*(?=\))', line)
    if r:
        return int(r.group())
    else:
        return 1


def extract_overlapping_memory_access(line: str) -> int:
    match = re.search('(?<=OVERLAPPING_MEMORY_ACCESSES\()\d*(?=\))', line)
    if match:
        return int(match.group())
    else:
        return 0


def extract_lambda_for_index_calculations(parameters: List[str], start_index):
    if start_index:
        f = 'INDIRECT_ACCESS_RANGE_START('
    else:
        f = 'INDIRECT_ACCESS_RANGE_END('
    for p in parameters:
        s = p.find(f)
        if s != -1:
            s += len(f)
            return p[s:]
    print('Error: Could not find one of lambdas for index calculations.')
    sys.exit(1)


def utils_extract_function_parameters(start_position_in_code: int, code: str) -> List[str]:
    parameters = []
    s = e = start_position_in_code
    unmatched_open_braces = 0
    while code[e if e == 0 else e - 1] != ')' or unmatched_open_braces != 0:
        if code[e] == ')':
            unmatched_open_braces -= 1
        elif code[e] == '(':
            if unmatched_open_braces == 0:
                s = e + 1
            unmatched_open_braces += 1
        if unmatched_open_braces == 1 and code[e] == ',':
            parameters.append(code[s:e].strip())
            s = e + 1
        e += 1
    parameters.append(code[s:e-1].strip())
    return parameters


def extract_buffer_info(buffers: List[buffer_spec.Buffer], code: str):
    device_hopper_malloc = 'device_hopper::malloc('
    device_hopper_aligned_malloc = 'device_hopper::aligned_malloc('
    device_hopper_reuse_existing_buffer = 'device_hopper::use_existing_buffer('

    buffer_types = {}
    buffer_element_counts = {}
    buffers_on_the_heap = []
    for line in code.split('\n'):
        if device_hopper_malloc in line or \
           device_hopper_reuse_existing_buffer in line or \
           device_hopper_aligned_malloc in line:
            # This branch extracts the types and sizes of the buffers
            # that will be transferred to and/or from the GPU.
            # Get the buffer type
            split_line = line.split()
            if split_line[0] == 'struct' or split_line[0] == 'unsigned':
                split_line[0] = split_line[0] + ' ' + split_line[1]
                del split_line[1]
            type = split_line[0].strip()
            buffer_name = split_line[1].strip().replace('*', '').strip()
            buffer_types[buffer_name] = type

            # Get the element count
            if device_hopper_malloc in line:
                s = line.find(device_hopper_malloc)
                s += len(device_hopper_malloc)
            elif device_hopper_aligned_malloc in line:
                s = line.find(device_hopper_aligned_malloc)
                s += len(device_hopper_aligned_malloc)
            elif device_hopper_reuse_existing_buffer in line:
                s = line.find(device_hopper_reuse_existing_buffer)
                s += len(device_hopper_reuse_existing_buffer)
            else:
                print('Error: This branch should never be executed.')
                sys.exit(1)
            e = s
            while line[e] != ')':
                e += 1
            parameters = line[s:e].split(',')
            if device_hopper_aligned_malloc in line:
                buffer_element_counts[buffer_name] = parameters[1].strip()
            else:
                buffer_element_counts[buffer_name] = parameters[0].strip()

            # Remember that this buffer is on the heap. We will use this information in the next branch.
            buffers_on_the_heap.append(buffer_name)

    # This block extracts access direction, access pattern and other optional information about the buffers.
    # TODO: Refactor
    # 1. Split this function in two functions.
    # 2. The code that finds the matches and searches for the end of the function calls
    #    is potentially duplicated.
    buffer_decl = 'device_hopper::buf'
    interim_results_property = 'data_kind::INTERIM_RESULTS'
    matches = re.finditer(buffer_decl, code)
    for match in matches:
        # Find the parameter list of the buffer declaration
        parameters = utils_extract_function_parameters(match.start(), code)

        #line = code[s:e - 1]
        #parameters = [param.strip() for param in line.split(',')]

        # Extract the access pattern, direction, the buffer name, and optional implementation specifications
        data_kind = None
        if parameters[2] == interim_results_property:
            data_kind      = buffer_spec.data_kind.INTERIM_RESULTS
            access_pattern = access_pattern_str_to_enum(parameters[3])
        else:
            access_pattern   = access_pattern_str_to_enum(parameters[2])
        access_direction = access_direction_str_to_enum(parameters[1])
        buffer_name      = parameters[0].strip()
        use_texture = False
        for p in parameters:
            if p.find('gpu_implementation::TEXTURE') != -1:
                use_texture = True

        if buffer_name in buffers_on_the_heap:
            # If we enter this branch then the buffer is on the heap and we already have all information
            # that we need to create a 'Buffer' object.
            b = buffer_spec.Buffer(
                access_pattern,
                access_direction,
                buffer_spec.address_space.GLOBAL,
                buffer_name,
                buffer_element_counts[buffer_name],
                buffer_types[buffer_name])
            if data_kind != None:
                b.set_data_kind(data_kind)
        else:
            # If we enter this branch then the buffer is on the stack and
            # we do not know the type and the number of elements yet.
            print('Error: Buffers on the stack are not supported yet.')
            sys.exit(1)

        # Extract how many elements are accessed per thread id.
        if access_pattern == buffer_spec.access_pattern.BY_THREAD_ID:
            ae = extract_number_of_elements_accessed_per_thread_id(line)
            b.set_number_of_elements_accessed_per_block_or_thread_id(ae)
            oa = extract_overlapping_memory_access(line)
            b.set_number_of_overlapping_memory_accesses(oa)
        elif access_pattern == buffer_spec.access_pattern.INDIRECT:
            buffer_with_indices = parameters[-1].replace(')','')
            start_index_lambda  = extract_lambda_for_index_calculations(parameters, start_index=True)
            final_index_lambda  = extract_lambda_for_index_calculations(parameters, start_index=False)
            b.set_intermediate_buffer_for_indirect_accesses(buffer_with_indices)
            b.set_lambda_to_compute_start_index_for_indirect_accesses(start_index_lambda)
            b.set_lambda_to_compute_final_index_for_indirect_accesses(final_index_lambda)
        elif access_pattern == buffer_spec.access_pattern.CONTINUOUS:
            access_pattern_parameters = utils_extract_function_parameters(0, parameters[2 if data_kind == None else 3])
            start_index_lambda = utils_extract_function_parameters(0, access_pattern_parameters[0])[0]
            end_index_lambda   = utils_extract_function_parameters(0, access_pattern_parameters[1])[0]
            b.set_lambda_to_compute_start_index_for_continuous_accesses(start_index_lambda)
            b.set_lambda_to_compute_final_index_for_continuous_accesses(end_index_lambda)
        elif access_pattern == buffer_spec.access_pattern.SUCCESSIVE_SUBSECTIONS:
            ae = extract_number_of_elements_accessed_per_batch(parameters[2])
            b.set_number_of_elements_accessed_per_block_or_thread_id(ae)
        b.set_use_texture_on_gpu(use_texture)

        buffers.append(b)


def extract_indirect_accesses(buffers: List[buffer_spec.Buffer]):
    for checked_b in buffers:
        for current_b in buffers:
            if checked_b.lambda_to_compute_final_index_for_continuous_accesses != None:
                if checked_b.lambda_to_compute_final_index_for_continuous_accesses.find(current_b.buffer_name) != -1:
                    checked_b.set_is_accessed_indirectly(True)
                    current_b.set_contains_indices(True)


def extract_iteration_space_bounds(code: str, is_2d_kernel: bool):
    parallel_for_start = 'parallel_for pf('
    search_end = 'DEVICE_HOPPER_LAMBDA'
    s = code.find(parallel_for_start)
    parameters = utils_extract_function_parameters(s, code)

    if is_2d_kernel:
        starts = utils_extract_function_parameters(0, parameters[0])
        ends = utils_extract_function_parameters(0, parameters[1])
        return starts, ends
    else:
        return (parameters[0].strip(), 0), (parameters[1].strip(), 1)


def extract_scalar_parameters(code: str) -> List[str]:
    scalar_parameters_start = 'pf.add_scalar_parameters('
    s = code.find(scalar_parameters_start)
    if s == -1:
        return []
    s += len(scalar_parameters_start)
    e = s
    while code[e] != ')':
        e += 1
    parameters = code[s:e].split(',')
    return [p.strip() for p in parameters]


def extract_scalar_parameter_types(code: str, scalar_parameters: List[str]) -> Dict[str, str]:
    parameter_types = {}
    for l in code.split('\n'):
        for sp in scalar_parameters:
            if l.find(sp) != -1:
                split = l.strip().split(' ')
                if split[0] == 'struct' or split[0] == 'unsigned':
                    split[0] = split[0] + ' ' + split[1]
                    del split[1]
                potential_type = split[0].strip()
                if potential_type in STANDARD_TYPES and not sp in parameter_types.keys():
                    parameter_types[sp] = potential_type
    return parameter_types


def is_cuda_file(file_name: str) -> bool:
    return file_name[-3:] == '.cu'


def read_the_template() -> str:
    f = open('template.h', 'r')
    t = f.read()
    f.close()
    return t


def read_the_opencl_template() -> str:
    f = open('opencl_kernel_template.cl')
    t = f.read()
    f.close()
    return t


def insert_the_block_size(block_size, block_size_is_set: bool, code: str) -> str:
    device_hopper_core_include = '#include "../device_hopper/core.h"'
    if block_size_is_set:
        # insert BLOCK_SIZE
        # if is_2d_kernel:
        if math.log(block_size[0], 2) % 1 != 0 or math.log(block_size[1], 2) % 1 != 0:
            print('Error: The block sizes are not a power of 2')
            exit(1)
        defines = '#define BLOCK_SIZE_X ' + str(block_size[0]) + '\n' + \
                  '#define BLOCK_SIZE_Y ' + str(block_size[1]) + '\n' + \
                  '#define LOG2_BLOCK_SIZE_X ' + str(int(math.log(block_size[0], 2))) + '\n' + \
                  '#define LOG2_BLOCK_SIZE_Y ' + str(int(math.log(block_size[1], 2))) + '\n'
        code = code.replace(device_hopper_core_include, defines + device_hopper_core_include)
        '''else:
            code =  code.replace(
                device_hopper_core_include,
                '#define BLOCK_SIZE ' + str(block_size) + 'u\n' + device_hopper_core_include)
            # insert LOG2_BLOCK_SIZE
            if math.log(block_size, 2) % 1 != 0:
                print('Error: The block size is not a power of 2')
                exit(1)
            code = code.replace(
                device_hopper_core_include,
                '#define LOG2_BLOCK_SIZE ' + str(int(math.log(block_size, 2))) + '\n' + device_hopper_core_include)'''
    else:
        print('Error: The block size is not set and this is not a reduction')
        print('The translator cannot automatically choose a block size yet except for reductions.')
        sys.exit(1)
    return code


def insert_misc_defines(
        buffers: List[buffer_spec.Buffer],
        code: str,
        is_2d_kernel: bool,
        is_idempotent: bool,
        is_reduction: bool,
        preferred_device: Device) -> str:
    device_hopper_core_include = '#include "../device_hopper/core.h"'
    # define SOURCE_TO_SOURCE_TRANSLATED
    code = code.replace(device_hopper_core_include, '#define SOURCE_TO_SOURCE_TRANSLATED\n' + device_hopper_core_include)
    # define PLASTIC
    code = code.replace(device_hopper_core_include, '#define PLASTIC\n' + device_hopper_core_include)
    # define 1D_KERNEL or 2D_KERNEL
    if is_2d_kernel:
        code = code.replace(device_hopper_core_include, '#define KERNEL_IS_2D\n' + device_hopper_core_include)
    else:
        code = code.replace(device_hopper_core_include, '#define KERNEL_IS_1D\n' + device_hopper_core_include)
    # define OpenCL
    if not is_reduction:
        code = code.replace(device_hopper_core_include, '#define OPENCL\n' + device_hopper_core_include)
    else:
        code = code.replace(device_hopper_core_include, '#define OMP\n' + device_hopper_core_include)
        code = code.replace(device_hopper_core_include, '#define IS_REDUCTION\n' + device_hopper_core_include)
    # define SLICED_DATA_TRANSFERS
    if preferred_device == Device.CPU:
        for b in buffers:
            if b.access_pattern != buffer_spec.access_pattern.RANDOM:
                code = code.replace(device_hopper_core_include, '#define SLICED_DATA_TRANSFERS\n' + device_hopper_core_include)
                break
    # define ABANDON_SLICES
    if is_idempotent and preferred_device == Device.GPU:
        code = code.replace(device_hopper_core_include, '#define ABANDON_SLICES\n' + device_hopper_core_include)
    return code


def insert_header_file(header_file: str, code: str) -> str:
    device_hopper_main = 'DEVICE_HOPPER_MAIN'
    return code.replace(device_hopper_main, '#include "' + header_file + '"\n' + device_hopper_main)


def insert_buffer_related_function_parameter_declarations(buffers: List[buffer_spec.Buffer], template: str):
    buffer_decls               = ''
    buffer_element_count_decls = ''
    buffer_element_size_decls  = ''
    forward_decls              = ''
    element_counts_set         = set()
    forward_decls_set          = set()
    for b in buffers:
        buffer_decls += b.type + ' *' + b.buffer_name + ', '
        if not b.type.replace('unsigned', '').replace('signed', '').strip() in STANDARD_TYPES and \
           not b.type in forward_decls_set:
            forward_decls += b.type + ';\n'
            forward_decls_set.add(b.type)

        if not b.element_count in element_counts_set:
            buffer_element_count_decls += 'size_t ' + b.element_count + ', '
            element_counts_set.add(b.element_count)
        #buffer_element_size_decls += 'size_t ' + b.buffer_name + '_element_size' + ', '

    template = template.replace('//%BUFFER_PARAMETER_DECLS%',                buffer_decls)
    template = template.replace('//%BUFFER_ELEMENT_COUNTS_PARAMETER_DECLS%', buffer_element_count_decls)
    template = template.replace('//%BUFFER_ELEMENT_SIZES_PARAMETER_DECLS%',  buffer_element_size_decls)
    template = template.replace('//%FORWARD_DECLARATION%',                   '')
    return template


def insert_buffer_related_function_parameters(buffers: List[buffer_spec.Buffer], template: str):
    buffer_parameters = ''
    buffer_element_count_parameters = ''
    buffer_element_size_parameters = ''
    element_count_set = set()
    for b in buffers:
        buffer_parameters += b.buffer_name + ', '
        if not b.element_count in element_count_set:
            buffer_element_count_parameters += b.element_count + ', '
            element_count_set.add(b.element_count)
        #buffer_element_size_parameters += b.buffer_name + '_element_size,'

    template = template.replace('//%BUFFERS%', buffer_parameters)
    template = template.replace('//%BUFFER_ELEMENT_COUNTS%', buffer_element_count_parameters)
    template = template.replace('//%BUFFER_ELEMENT_SIZES%', buffer_element_size_parameters)
    return template


def insert_parallel_for(
        buffers,
        code,
        is_reduction,
        iteration_space_size_start_x,
        iteration_space_size_start_y,
        iteration_space_size_end_x,
        iteration_space_size_end_y,
        scalar_parameters):
    parallel_for_start = 'parallel_for'
    parallel_for_end   = 'run();'

    s = code.find(parallel_for_start)
    e = code.find(parallel_for_end)
    e += len(parallel_for_end)

    # Parallel for call
    parallel_for_replacement = 'plasticity::kernels::parallel_for(%PARAMETERS%);'
    # Assemble parameters for the parallel_for call
    if is_reduction:
        parameters = ''
    else:
        parameters = '\n#if defined(EXECUTABLE)\ncl_handle,\n#else\ncpu_context,\ncpu_queue,\ncpu_handle,\n#endif\n'
    parameters += str(iteration_space_size_start_x) + ',\n' + \
                  str(iteration_space_size_start_y) + ',\n' + \
                  str(iteration_space_size_end_x) + ',\n' + \
                  str(iteration_space_size_end_y) + ',\n'
    for b in buffers:
        parameters += b.buffer_name + ',\n'
    buffer_element_counts_set = set()
    for b in buffers:
        if not b.element_count in buffer_element_counts_set:
            parameters += b.element_count + ',\n'
            buffer_element_counts_set.add(b.element_count)
    #for b in buffers:
    #    parameters += 'sizeof(' + b.type + '),\n'
    for sp in scalar_parameters:
        if not scalar_parameter_is_buffer_size(buffers, sp):
            parameters += sp + ',\n'
    parameters += 'context'
    parallel_for_replacement = parallel_for_replacement.replace('%PARAMETERS%', parameters)
    return code[:s] + 'cudaDeviceSynchronize();\n' + parallel_for_replacement + code[e:]
    #return code[:s] + parallel_for_replacement + code[e:]


def insert_boiler_plate_code_for_textures(buffers: List[buffer_spec.Buffer], t: str):
    f = open('boiler_plate_code_for_textures_template.h', 'r')
    boiler_plate_code_template = f.read()
    f.close()

    code = ''
    for b in buffers:
        if b.use_texture_on_gpu:
            code += boiler_plate_code_template.replace('%BUFFER_NAME%', b.buffer_name)\
                                              .replace('%TYPE%', b.type)
    return t.replace('//%BOILER_PLATE_CODE_FOR_TEXTURES%', code)


def insert_parameters_for_indirect_buffer_accesses(buffers: List[buffer_spec.Buffer], template: str) -> str:
    parameters = ''
    parameter_decls = ''
    for b in buffers:
        if b.contains_indices:
            parameters += b.buffer_name + ','
            parameter_decls += b.type + ' *' + b.buffer_name + ','
    template = template.replace('//%BUFFERS_CONTAINING_INDICES%', parameters)\
                       .replace('//%BUFFERS_CONTAINING_INDICES_DECLS%', parameter_decls)
    return template


def insert_scalar_parameter_declarations(
        buffers: List[buffer_spec.Buffer],
        scalar_parameters: List[str],
        scalar_parameter_types: Dict[str, str],
        template: str):
    parameter_decl = ''
    for sp in scalar_parameters:
        if not scalar_parameter_is_buffer_size(buffers, sp):
            parameter_decl += scalar_parameter_types[sp] + ' ' + sp + ', '
    return template.replace('//%SCALAR_PARAMETER_DECLS%', parameter_decl)


def insert_scalar_parameters(
        buffers: List[buffer_spec.Buffer],
        scalar_parameters: List[str],
        scalar_parameter_types: Dict[str, str],
        template: str):
    parameters = ''
    for sp in scalar_parameters:
        if not scalar_parameter_is_buffer_size(buffers, sp):
            if scalar_parameter_types[sp].find('&') == -1:
                parameters += sp + ',\n'
            else:
                parameters += 'std::ref(' + sp + '),\n'

    return template.replace('//%SCALAR_PARAMETERS%', parameters)


def insert_opencl_buffer_handles(buffers: List[buffer_spec.Buffer], template: str):
    buffer_handles = ''
    for b in buffers:
        buffer_handles += 'cl_mem ' + b.buffer_name + '_d = NULL;\n'
    return template.replace('//%OPENCL_BUFFER_HANDLES%', buffer_handles)


def insert_cuda_buffer_handles(buffers: List[buffer_spec.Buffer], template: str):
    buffer_handles = ''
    for b in buffers:
        buffer_handles += b.type + ' *' + b.buffer_name + '_d = NULL;\n'
        if b.is_accessed_indirectly:
            buffer_handles += 'size_t ' + b.buffer_name + '_d_size = 0;\n'
    return template.replace('//%CUDA_BUFFER_HANDLES%', buffer_handles)


def insert_cuda_buffer_creation(indices_are_used_only_for_memory_accesses: bool, buffers: List[buffer_spec.Buffer], template: str):
    cuda_malloc_template = 'if (cudaMalloc(&%POINTER%, %SIZE_IN_BYTES%) != cudaSuccess)\n'\
                           'utils::exit_with_err("Could not allocate device buffers");\n'
    malloc_code = ''
    for b in buffers:
        if b.address_space == buffer_spec.address_space.GLOBAL and not b.is_accessed_indirectly:
            if b.access_pattern == buffer_spec.access_pattern.RANDOM or \
               b.access_pattern == buffer_spec.access_pattern.REDUCTION_INTERMEDIATE_RESULTS or \
               (not indices_are_used_only_for_memory_accesses and
                not b.access_pattern == buffer_spec.access_pattern.REDUCTION): # or any_of_the_buffers_is_random_access:
                pointer      = 'h.cuda.' + b.buffer_name + '_d'
                size         = b.element_count + ' * ' + ' sizeof(' + b.type + ')' #+ b.buffer_name + '_element_size'
                malloc_code += cuda_malloc_template.replace('%POINTER%', pointer).replace('%SIZE_IN_BYTES%', size)
            elif b.access_pattern == buffer_spec.access_pattern.BY_THREAD_ID:
                pointer      = 'h.cuda.' + b.buffer_name + '_d'
                size         = '(slice_size * BLOCK_SIZE * ' + \
                               str(b.number_of_elements_accessed_per_block_or_thread_id - b.number_of_overlapping_accesses) + \
                               ' + ' + str(b.number_of_overlapping_accesses) + ') * ' + ' sizeof(' + b.type + ')' #b.buffer_name + '_element_size'
                malloc_code += cuda_malloc_template.replace('%POINTER%', pointer).replace('%SIZE_IN_BYTES%', size)
            elif b.access_pattern == buffer_spec.access_pattern.BY_BLOCK_ID:
                pointer = 'h.cuda.' + b.buffer_name + '_d'
                # only one element is needed ber block (based on btree-find-k)
                size = 'slice_size * ' + ' sizeof(' + b.type + ')' #+ b.buffer_name + '_element_size'
                malloc_code += cuda_malloc_template.replace('%POINTER%', pointer) \
                    .replace('%SIZE_IN_BYTES%', size)
            elif b.access_pattern == buffer_spec.access_pattern.SUCCESSIVE_SUBSECTIONS:
                pointer = 'h.cuda.' + b.buffer_name + '_d'
                size    = 'slice_size * ' + str(b.number_of_elements_accessed_per_block_or_thread_id) + \
                          ' * sizeof(' + b.type + ')'
                malloc_code += cuda_malloc_template.replace('%POINTER%', pointer) \
                    .replace('%SIZE_IN_BYTES%', size)
            elif b.access_pattern == buffer_spec.access_pattern.INDIRECT:
                pass
                # Mallocs for buffers with indirect accesses are performed where the data transfers are performed.
                # This is so because new buffers need to be allocated dynamically in response to dynamically varying
                # on-device buffer size requirements.
            elif b.access_pattern == buffer_spec.access_pattern.REDUCTION:
                pointer = 'h.cuda.' + b.buffer_name + '_d'
                size = 'slice_size * ' + ' sizeof(' + b.type + ')' #+ b.buffer_name + '_element_size'
                malloc_code += cuda_malloc_template.replace('%POINTER%', pointer).replace('%SIZE_IN_BYTES%', size)
            elif b.access_pattern == buffer_spec.access_pattern.CONTINUOUS:
                continuous_access_malloc_template_file = open('continuous_access_malloc_template.h', 'r')
                continuous_access_malloc_template = continuous_access_malloc_template_file.read()
                continuous_access_malloc_template_file.close()

                malloc_code += continuous_access_malloc_template\
                    .replace('%START_INDEX_LAMBDA%', b.lambda_to_compute_start_index_for_continuous_accesses)\
                    .replace('%FINAL_INDEX_LAMBDA%', b.lambda_to_compute_final_index_for_continuous_accesses)\
                    .replace('%BUFFER_NAME%', b.buffer_name) \
                    .replace('%ELEMENT_SIZE%', 'sizeof(' + b.type + ')')
            else:
                print('Error: Unknown access pattern')
                traceback.print_stack()
                exit(1)

            if b.use_texture_on_gpu:
                f = open('code_to_bind_cuda_textures_template.h', 'r')
                code_to_bind_textures = f.read()
                f.close()
                code_to_bind_textures += '\n'
                malloc_code += code_to_bind_textures.replace('%BUFFER_NAME%', b.buffer_name)\
                                                    .replace('%ELEMENT_COUNT%', b.element_count)
    return template.replace('//%CUDA_MALLOC%', malloc_code)


def insert_cleanup_code(buffers: List[buffer_spec.Buffer], template: str):
    # Insert OpenCL cleanup code
    opencl_cleanup_code = ''
    opencl_template = 'if (h->opencl.%POINTER%) {\n' \
                        'err |= clReleaseMemObject(h->opencl.%POINTER%);\n' \
                        'h->opencl.%POINTER% = NULL;\n' \
                      '}\n'
    for b in buffers:
        opencl_cleanup_code += opencl_template.replace('%POINTER%', b.buffer_name + '_d')
    template = template.replace('//%OPENCL_RELEASE_MEM_OBJECT%', opencl_cleanup_code)

    # Insert CUDA cleanup code
    cuda_cleanup_code = ''
    cuda_template = 'if (h->cuda.%POINTER%) {\n' \
                      'if (cudaFree(h->cuda.%POINTER%) != cudaSuccess) utils::exit_with_err("Could not free recordsD");\n' \
                      'h->cuda.%POINTER% = NULL;\n' \
                    '}\n'
    for b in buffers:
        cuda_cleanup_code += cuda_template.replace('%POINTER%', b.buffer_name + '_d')
    template = template.replace('//%CUDA_RELEASE_MEM_OBJECT%', cuda_cleanup_code)
    return template


def insert_code_for_data_transfers_to_the_gpu(indices_are_used_only_for_memory_accesses: bool,
                                              is_reduction: bool,
                                              buffers: List[buffer_spec.Buffer],
                                              preferred_device: Device,
                                              template: str):
    placeholder_for_buffer_sizes = '//%BUFFER_SIZES_OF_INTERMEDIATE_BUFFERS_FOR_INDIRECT_MEMORY_ACCESSES%'

    at_least_one_input_buffer_is_not_random_access = False
    for b in buffers:
        if b.access_pattern != buffer_spec.access_pattern.RANDOM:
            if b.access_direction == buffer_spec.access_direction.IN:
                at_least_one_input_buffer_is_not_random_access  = True
            elif b.access_direction == buffer_spec.access_direction.OUT:
                pass
            elif b.access_direction == buffer_spec.access_direction.INOUT:
                at_least_one_input_buffer_is_not_random_access  = True
            else:
                print('Error: Unknown access direction.')
                sys.exit(1)
    if (at_least_one_input_buffer_is_not_random_access and indices_are_used_only_for_memory_accesses) or is_reduction:
        f = open('transfer_data_to_device_function_stub_template.h', 'r')
        data_transfer_function_stub = f.read()
        f.close()

        f = open('transfer_data_to_device_function_call_template.h', 'r')
        data_transfer_function_call = f.read()
        f.close()

        template = template\
            .replace('//%TRANSFER_DATA_TO_DEVICE_FUNCTION_CALL%', data_transfer_function_call)\
            .replace('//%TRANSFER_DATA_TO_DEVICE_FUNCTION_STUB%', data_transfer_function_stub)
    else:
        template = template \
            .replace('//%TRANSFER_DATA_TO_DEVICE_FUNCTION_CALL%', '') \
            .replace('//%TRANSFER_DATA_TO_DEVICE_FUNCTION_STUB%', '')

    if preferred_device == Device.GPU:
        data_transfer_template_for_non_random_access_buffers_file = open(
            'non_chunked_data_transfer_template_for_non_random_access_buffers.h', 'r')
        data_transfer_template_for_non_random_access_buffers = data_transfer_template_for_non_random_access_buffers_file.read()
        data_transfer_template_for_non_random_access_buffers_file.close()

        data_transfer_template_for_random_access_buffers_file = open(
            'non_chunked_data_transfer_template_for_random_access_buffers.h', 'r')
        data_transfer_template_for_random_access_buffers = data_transfer_template_for_random_access_buffers_file.read()
        data_transfer_template_for_random_access_buffers_file.close()

        data_transfer_template_for_continuous_access_buffers_file = open(
            'non_chunked_data_transfer_template_for_continuous_access_buffers.h', 'r')
        data_transfer_template_for_continuous_access_buffers = data_transfer_template_for_continuous_access_buffers_file.read()
        data_transfer_template_for_continuous_access_buffers_file.close()
    else:
        data_transfer_template_for_non_random_access_buffers_file = open(
            'chunked_data_transfer_template_for_non_random_access_buffers.h', 'r')
        data_transfer_template_for_non_random_access_buffers = data_transfer_template_for_non_random_access_buffers_file.read()
        data_transfer_template_for_non_random_access_buffers_file.close()

        data_transfer_template_for_random_access_buffers_file = open(
            'chunked_data_transfer_template_for_random_access_buffers.h', 'r')
        data_transfer_template_for_random_access_buffers = data_transfer_template_for_random_access_buffers_file.read()
        data_transfer_template_for_random_access_buffers_file.close()

        data_transfer_template_for_continuous_access_buffers_file = open(
            'chunked_data_transfer_template_for_continuous_access_buffers.h', 'r')
        data_transfer_template_for_continuous_access_buffers = data_transfer_template_for_continuous_access_buffers_file.read()
        data_transfer_template_for_continuous_access_buffers_file.close()

    code_for_per_slice_data_transfers = ''
    code_for_one_time_data_transfers  = ''
    code_for_one_time_data_transfers_inside_the_while_loop = '' # This is only used for random access output buffers if
                                                                # the GPU is the preferred device.
    for b in buffers:
        if b.access_direction == buffer_spec.access_direction.IN or \
           b.access_direction == buffer_spec.access_direction.INOUT or \
           (b.access_direction == buffer_spec.access_direction.OUT and b.access_pattern == buffer_spec.access_pattern.RANDOM):
            if b.access_pattern == buffer_spec.access_pattern.RANDOM or \
               (not indices_are_used_only_for_memory_accesses and
                not b.access_pattern == buffer_spec.access_pattern.REDUCTION_INTERMEDIATE_RESULTS and
                not b.access_pattern == buffer_spec.access_pattern.REDUCTION):
                code = data_transfer_template_for_random_access_buffers\
                    .replace('%ADDITIONAL_ELEMENTS_FOR_THE_THREAD_WITH_THE_HIGHEST_ID_IN_EACH_BLOCK%',
                             str(b.number_of_overlapping_accesses)) \
                    .replace('%BUFFER_ELEMENT_COUNT%', b.element_count) \
                    .replace('%DESTINATION_BUFFER%', 'h.cuda.' + b.buffer_name + '_d')\
                    .replace('%ELEMENT_SIZE%', 'sizeof(' + b.type + ')')\
                    .replace('%DESTINATION_POINTER%', 'd_ptr') \
                    .replace('%SOURCE_BUFFER%', b.buffer_name) \
                    .replace('%SOURCE_POINTER%', 'h_ptr')\
                    .replace('%TRANSFER_DIRECTION%', 'cudaMemcpyHostToDevice')
                if b.access_direction == buffer_spec.access_direction.OUT and preferred_device == Device.GPU:
                    code_for_one_time_data_transfers_inside_the_while_loop += code
                else:
                    code_for_one_time_data_transfers += code
            elif b.access_pattern == buffer_spec.access_pattern.REDUCTION_INTERMEDIATE_RESULTS:
                data_transfer_template_for_random_access_buffers_file = open(
                    'non_chunked_data_transfer_template_for_random_access_buffers.h', 'r')
                data_transfer_template_for_random_access_buffers = data_transfer_template_for_random_access_buffers_file.read()
                data_transfer_template_for_random_access_buffers_file.close()

                code = data_transfer_template_for_random_access_buffers\
                    .replace('%ADDITIONAL_ELEMENTS_FOR_THE_THREAD_WITH_THE_HIGHEST_ID_IN_EACH_BLOCK%',
                             str(b.number_of_overlapping_accesses)) \
                    .replace('%BUFFER_ELEMENT_COUNT%', b.element_count) \
                    .replace('%DESTINATION_BUFFER%', 'h.cuda.' + b.buffer_name + '_d')\
                    .replace('%ELEMENT_SIZE%', 'sizeof(' + b.type + ')')\
                    .replace('%DESTINATION_POINTER%', 'd_ptr') \
                    .replace('%SOURCE_BUFFER%', b.buffer_name) \
                    .replace('%SOURCE_POINTER%', 'h_ptr')\
                    .replace('%TRANSFER_DIRECTION%', 'cudaMemcpyHostToDevice')
                code_for_one_time_data_transfers += code

            elif b.access_pattern == buffer_spec.access_pattern.BY_THREAD_ID:
                code_for_per_slice_data_transfers += data_transfer_template_for_non_random_access_buffers\
                    .replace('%BUFFER_NAME_HOST%', b.buffer_name)\
                    .replace('%ADDITIONAL_ELEMENTS_FOR_THE_THREAD_WITH_THE_HIGHEST_ID_IN_EACH_BLOCK%',
                             str(b.number_of_overlapping_accesses))\
                    .replace('%OFFSET_SHIFT%', 'BLOCK_SIZE_X')\
                    .replace('%BUFFER_NAME_DEVICE%', b.buffer_name + '_d')\
                    .replace('%ELEMENT_SIZE%', 'sizeof(' + b.type + ')')\
                    .replace('%DESTINATION_POINTER%', 'd_ptr')\
                    .replace('%NUMBER_OF_ELEMENTS_ACCESSED_PER_BLOCK_OR_THREAD_ID%',
                             str(b.number_of_elements_accessed_per_block_or_thread_id -
                                 b.number_of_overlapping_accesses)) \
                    .replace('%SOURCE_POINTER%', 'h_ptr')\
                    .replace('%TRANSFER_DIRECTION%', 'cudaMemcpyHostToDevice')
            elif b.access_pattern == buffer_spec.access_pattern.BY_BLOCK_ID:
                code_for_per_slice_data_transfers += data_transfer_template_for_non_random_access_buffers \
                    .replace('%ADDITIONAL_ELEMENTS_FOR_THE_THREAD_WITH_THE_HIGHEST_ID_IN_EACH_BLOCK%',
                             str(b.number_of_overlapping_accesses)) \
                    .replace('%BUFFER_NAME_HOST%', b.buffer_name) \
                    .replace('%BUFFER_NAME_DEVICE%', b.buffer_name + '_d') \
                    .replace('%DESTINATION_POINTER%', 'd_ptr') \
                    .replace('%ELEMENT_SIZE%', 'sizeof(' + b.type + ')')\
                    .replace('%NUMBER_OF_ELEMENTS_ACCESSED_PER_BLOCK_OR_THREAD_ID%',
                             str(b.number_of_elements_accessed_per_block_or_thread_id)) \
                    .replace('%OFFSET_SHIFT%', '1') \
                    .replace('%SOURCE_POINTER%', 'h_ptr')\
                    .replace('%TRANSFER_DIRECTION%', 'cudaMemcpyHostToDevice')
            elif b.access_pattern == buffer_spec.access_pattern.SUCCESSIVE_SUBSECTIONS:
                ea = str(b.number_of_elements_accessed_per_block_or_thread_id)
                code_for_per_slice_data_transfers += data_transfer_template_for_non_random_access_buffers \
                    .replace('%ADDITIONAL_ELEMENTS_FOR_THE_THREAD_WITH_THE_HIGHEST_ID_IN_EACH_BLOCK%', '0') \
                    .replace('%BUFFER_NAME_HOST%', b.buffer_name) \
                    .replace('%BUFFER_NAME_DEVICE%', b.buffer_name + '_d') \
                    .replace('%DESTINATION_POINTER%', 'd_ptr') \
                    .replace('%ELEMENT_SIZE%', 'sizeof(' + b.type + ')') \
                    .replace('%NUMBER_OF_ELEMENTS_ACCESSED_PER_BLOCK_OR_THREAD_ID%', ea) \
                    .replace('%OFFSET_SHIFT%', '1') \
                    .replace('%SOURCE_POINTER%', 'h_ptr') \
                    .replace('%TRANSFER_DIRECTION%', 'cudaMemcpyHostToDevice')
            elif b.access_pattern == buffer_spec.access_pattern.CONTINUOUS:
                buffer_resizing_code = ''
                if b.is_accessed_indirectly: #and not any_of_the_buffers_is_random_access:
                    template_file = open('reallocate_indirectly_accessed_buffers_template.h', 'r')
                    buffer_resizing_code = template_file.read()
                    template_file.close()
                    buffer_resizing_code = buffer_resizing_code.replace('%BUFFER_NAME%', b.buffer_name)

                code_for_per_slice_data_transfers += data_transfer_template_for_continuous_access_buffers \
                    .replace('%BUFFER_NAME_HOST%', b.buffer_name) \
                    .replace('%BUFFER_NAME_DEVICE%', b.buffer_name + '_d') \
                    .replace('%ELEMENT_SIZE%', 'sizeof(' + b.type + ')') \
                    .replace('%DESTINATION_POINTER%', 'd_ptr')\
                    .replace('%SOURCE_POINTER%', 'h_ptr') \
                    .replace('%START_INDEX_LAMBDA%', b.lambda_to_compute_start_index_for_continuous_accesses)\
                    .replace('%FINAL_INDEX_LAMBDA%', b.lambda_to_compute_final_index_for_continuous_accesses)\
                    .replace('%TRANSFER_DIRECTION%', 'cudaMemcpyHostToDevice')\
                    .replace('//%CHECK_IF_INDIRECTLY_ACCESSED_BUFFERS_NEED_TO_BE_RESIZED%', buffer_resizing_code) \
                    .replace('%DEVICE_BUFFER_INDEX%', '0')  # 'start_index' if any_of_the_buffers_is_random_access else '0')\
            elif b.access_pattern == buffer_spec.access_pattern.INDIRECT:
                template = template.replace(placeholder_for_buffer_sizes,
                    'size_t ' + b.buffer_name + '_current_size_in_bytes = 0;\n' + placeholder_for_buffer_sizes)

                malloc_template_file = open('malloc_for_indirectly_accessed_buffers_template.h', 'r')
                malloc_template = malloc_template_file.read()
                malloc_template_file.close()

                malloc_template = malloc_template\
                    .replace('%START_INDEX_LAMBDA%', b.lambda_to_compute_start_index_for_indirect_accesses)\
                    .replace('%FINAL_INDEX_LAMBDA%', b.lambda_to_compute_final_index_for_indirect_accesses)\
                    .replace('%INTERMEDIATE_BUFFER%', b.intermediate_buffer_for_indirect_accesses)\
                    .replace('%BUFFER_NAME%', b.buffer_name)

                data_transfer_template_file = open('chunked_data_transfer_template_for_random_access_buffers.h', 'r')
                data_transfer_template = data_transfer_template_file.read()
                data_transfer_template_file.close()

                data_transfer_template = data_transfer_template\
                    .replace('%BUFFER_NAME_HOST%', '&' + b.buffer_name + '[indirect_start_index]')\
                    .replace('%BUFFER_ELEMENT_COUNT%', 'elements')\
                    .replace('%BUFFER_NAME_DEVICE%', b.buffer_name + '_d') \
                    .replace('%ELEMENT_SIZE%', 'sizeof(' + b.type + ')')\
                    .replace('%DESTINATION_POINTER%', 'd_ptr')\
                    .replace('%SOURCE_POINTER%', 'h_ptr')\
                    .replace('%TRANSFER_DIRECTION%', 'cudaMemcpyHostToDevice')

                code_for_per_slice_data_transfers += malloc_template.replace('%DATA_TRANSFER%', data_transfer_template)

            elif b.access_pattern == buffer_spec.access_pattern.REDUCTION:
                code_for_per_slice_data_transfers += data_transfer_template_for_non_random_access_buffers\
                    .replace('%BUFFER_NAME_HOST%', b.buffer_name) \
                    .replace('%ADDITIONAL_ELEMENTS_FOR_THE_THREAD_WITH_THE_HIGHEST_ID_IN_EACH_BLOCK%',
                             str(b.number_of_overlapping_accesses)) \
                    .replace('%OFFSET_SHIFT%', '1') \
                    .replace('%BUFFER_NAME_DEVICE%', b.buffer_name + '_d') \
                    .replace('%ELEMENT_SIZE%', 'sizeof(' + b.type + ')') \
                    .replace('%DESTINATION_POINTER%', 'd_ptr') \
                    .replace('%NUMBER_OF_ELEMENTS_ACCESSED_PER_BLOCK_OR_THREAD_ID%',
                             str(b.number_of_elements_accessed_per_block_or_thread_id)) \
                    .replace('%SOURCE_POINTER%', 'h_ptr') \
                    .replace('%TRANSFER_DIRECTION%', 'cudaMemcpyHostToDevice')
            else:
                print('Error: Unknown access pattern')
                traceback.print_stack()
                sys.exit(1)

    # Remove placeholders that will not be needed anymore
    template.replace(placeholder_for_buffer_sizes, '')

    # Add a variable for the return value of cudaMemcpy only if we use cudaMemcpy.
    if code_for_one_time_data_transfers != '':
        code_for_one_time_data_transfers = 'cudaError err;\n' + code_for_one_time_data_transfers

    # Return the results
    template = template.replace('//%DATA_TRANSFER_TO_DEVICE%', code_for_per_slice_data_transfers)\
                       .replace('//%ONE_TIME_DATA_TRANSFERS_TO_THE_GPU%', code_for_one_time_data_transfers)\
                       .replace('//%TRANSFER_RANDOM_ACCESS_OUTPUT_BUFFER_IF_THE_GPU_IS_THE_PREFERRED_DEVICE%',
                                code_for_one_time_data_transfers_inside_the_while_loop)
    return template


def insert_code_for_data_transfers_from_the_gpu(indices_are_used_only_for_memory_accesses: bool,
                                                is_reduction: bool,
                                                buffers: List[buffer_spec.Buffer],
                                                is_idempotent: bool,
                                                preferred_device: Device,
                                                template: str):
    if is_idempotent and preferred_device == Device.CPU:
        data_transfer_template_file = open('chunked_data_transfer_template_for_non_random_access_buffers.h', 'r')
    else:
        data_transfer_template_file = open('non_chunked_data_transfer_template_for_non_random_access_buffers.h', 'r')
    data_transfer_template = data_transfer_template_file.read()
    data_transfer_template_file.close()

    at_least_one_output_buffer_is_not_random_access = False
    for b in buffers:
        if b.access_pattern != buffer_spec.access_pattern.RANDOM:
            if b.access_direction == buffer_spec.access_direction.IN:
                pass
            elif b.access_direction == buffer_spec.access_direction.OUT:
                at_least_one_output_buffer_is_not_random_access = True
            elif b.access_direction == buffer_spec.access_direction.INOUT:
                at_least_one_output_buffer_is_not_random_access = True
            else:
                print('Error: Unknown access direction.')
                sys.exit(1)

    if (at_least_one_output_buffer_is_not_random_access and indices_are_used_only_for_memory_accesses) or is_reduction:
        f = open('transfer_data_from_device_function_stub_template.h', 'r')
        data_transfer_function_stub = f.read()
        f.close()

        f = open('transfer_data_from_device_function_call_template.h', 'r')
        data_transfer_function_call = f.read()
        f.close()

        template = template \
            .replace('//%TRANSFER_DATA_FROM_DEVICE_FUNCTION_CALL%', data_transfer_function_call) \
            .replace('//%TRANSFER_DATA_FROM_DEVICE_FUNCTION_STUB%', data_transfer_function_stub)
    else:
        template = template \
            .replace('//%TRANSFER_DATA_FROM_DEVICE_FUNCTION_CALL%', '') \
            .replace('//%TRANSFER_DATA_FROM_DEVICE_FUNCTION_STUB%', '')

    data_transfer_code = ''
    code_for_one_time_data_transfers = ''
    for b in buffers:
        if (b.access_direction == buffer_spec.access_direction.OUT or \
            b.access_direction == buffer_spec.access_direction.INOUT) and \
            b.data_kind        != buffer_spec.data_kind.INTERIM_RESULTS:
            if b.access_pattern == buffer_spec.access_pattern.RANDOM or \
               b.access_pattern == buffer_spec.access_pattern.REDUCTION_INTERMEDIATE_RESULTS or \
               (not indices_are_used_only_for_memory_accesses and
                not b.access_pattern == buffer_spec.access_pattern.REDUCTION_INTERMEDIATE_RESULTS and
                not b.access_pattern == buffer_spec.access_pattern.REDUCTION):
                code_for_one_time_data_transfers += cuda_memcpy_template.replace('%DESTINATION_BUFFER%', b.buffer_name) \
                    .replace('%SOURCE_BUFFER%', 'h.cuda.' + b.buffer_name + '_d') \
                    .replace('%ELEMENT_SIZE%', 'sizeof(' + b.type + ')') \
                    .replace('%ELEMENT_COUNT%', b.element_count) \
                    .replace('%TRANSFER_DIRECTION%', 'cudaMemcpyDeviceToHost')
            elif b.access_pattern == buffer_spec.access_pattern.BY_THREAD_ID:
                data_transfer_code += data_transfer_template.replace('%BUFFER_NAME_HOST%', b.buffer_name) \
                    .replace('%ADDITIONAL_ELEMENTS_FOR_THE_THREAD_WITH_THE_HIGHEST_ID_IN_EACH_BLOCK%',
                             str(b.number_of_overlapping_accesses)) \
                    .replace('%OFFSET_SHIFT%', 'BLOCK_SIZE_X')\
                    .replace('%BUFFER_NAME_DEVICE%', b.buffer_name + '_d') \
                    .replace('%ELEMENT_SIZE%', 'sizeof(' + b.type + ')')\
                    .replace('%DESTINATION_POINTER%', 'h_ptr') \
                    .replace('%NUMBER_OF_ELEMENTS_ACCESSED_PER_BLOCK_OR_THREAD_ID%',
                             str(b.number_of_elements_accessed_per_block_or_thread_id - b.number_of_overlapping_accesses)) \
                    .replace('%SOURCE_POINTER%', 'd_ptr') \
                    .replace('%TRANSFER_DIRECTION%', 'cudaMemcpyDeviceToHost')
            elif b.access_pattern == buffer_spec.access_pattern.BY_BLOCK_ID:
                data_transfer_code += data_transfer_template.replace('%BUFFER_NAME_HOST%', b.buffer_name) \
                    .replace('%ADDITIONAL_ELEMENTS_FOR_THE_THREAD_WITH_THE_HIGHEST_ID_IN_EACH_BLOCK%',
                             str(b.number_of_overlapping_accesses)) \
                    .replace('%OFFSET_SHIFT%', '1')\
                    .replace('%BUFFER_NAME_DEVICE%', b.buffer_name + '_d') \
                    .replace('%ELEMENT_SIZE%', 'sizeof(' + b.type + ')')\
                    .replace('%DESTINATION_POINTER%', 'h_ptr') \
                    .replace('%NUMBER_OF_ELEMENTS_ACCESSED_PER_BLOCK_OR_THREAD_ID%',
                             str(b.number_of_elements_accessed_per_block_or_thread_id)) \
                    .replace('%SOURCE_POINTER%', 'd_ptr') \
                    .replace('%TRANSFER_DIRECTION%', 'cudaMemcpyDeviceToHost')
            elif b.access_pattern == buffer_spec.access_pattern.SUCCESSIVE_SUBSECTIONS:
                ae = str(b.number_of_elements_accessed_per_block_or_thread_id)
                data_transfer_code += data_transfer_template.replace('%BUFFER_NAME_HOST%', b.buffer_name) \
                    .replace('%ADDITIONAL_ELEMENTS_FOR_THE_THREAD_WITH_THE_HIGHEST_ID_IN_EACH_BLOCK%', '0') \
                    .replace('%OFFSET_SHIFT%', '1') \
                    .replace('%BUFFER_NAME_DEVICE%', b.buffer_name + '_d') \
                    .replace('%ELEMENT_SIZE%', 'sizeof(' + b.type + ')') \
                    .replace('%DESTINATION_POINTER%', 'h_ptr') \
                    .replace('%NUMBER_OF_ELEMENTS_ACCESSED_PER_BLOCK_OR_THREAD_ID%', ae) \
                    .replace('%SOURCE_POINTER%', 'd_ptr') \
                    .replace('%TRANSFER_DIRECTION%', 'cudaMemcpyDeviceToHost')
            elif b.access_pattern == buffer_spec.access_pattern.CONTINUOUS:
                if is_idempotent and preferred_device == Device.CPU:
                    data_transfer_template_for_continuous_access_buffers_file = open(
                        'chunked_data_transfer_template_for_continuous_access_buffers.h', 'r')
                else:
                    data_transfer_template_for_continuous_access_buffers_file = open(
                        'non_chunked_data_transfer_template_for_continuous_access_buffers.h', 'r')
                data_transfer_template_for_continuous_access_buffers = data_transfer_template_for_continuous_access_buffers_file.read()
                data_transfer_template_for_continuous_access_buffers_file.close()

                data_transfer_code += data_transfer_template_for_continuous_access_buffers \
                    .replace('%BUFFER_NAME_HOST%', b.buffer_name) \
                    .replace('%BUFFER_NAME_DEVICE%', b.buffer_name + '_d') \
                    .replace('%ELEMENT_SIZE%', 'sizeof(' + b.type + ')') \
                    .replace('%DESTINATION_POINTER%', 'h_ptr') \
                    .replace('%SOURCE_POINTER%', 'd_ptr') \
                    .replace('%START_INDEX_LAMBDA%', b.lambda_to_compute_start_index_for_continuous_accesses) \
                    .replace('%FINAL_INDEX_LAMBDA%', b.lambda_to_compute_final_index_for_continuous_accesses) \
                    .replace('%TRANSFER_DIRECTION%', 'cudaMemcpyDeviceToHost')\
                    .replace('%DEVICE_BUFFER_INDEX%', '0') # 'start_index' if any_of_the_buffers_is_random_access else '0')
            else:
                print('Error: Unknown access pattern')
                traceback.print_stack()
                sys.exit(1)

    # Add a variable for the return value of cudaMemcpy only if we use cudaMemcpy.
    if code_for_one_time_data_transfers != '':
        code_for_one_time_data_transfers = 'cudaError err;\n' + code_for_one_time_data_transfers

    template = template.replace('//%DATA_TRANSFER_FROM_DEVICE%', data_transfer_code)\
                       .replace('//%ONE_TIME_DATA_TRANSFERS_FROM_THE_DEVICE%', code_for_one_time_data_transfers)
    return template


def insert_texture_object_instantiations(buffers: List[buffer_spec.Buffer], t: str) -> str:
    code = ''
    for b in buffers:
        if b.use_texture_on_gpu:
            code += b.buffer_name + '_tex_reader_struct ' + b.buffer_name + '_tex_reader;\n'
    code += '//%CUDA_KERNEL_CODE%'
    return t.replace('//%CUDA_KERNEL_CODE%', code)


def insert_cuda_kernel(
        any_of_the_buffers_is_random_access: bool,
        buffers: List[buffer_spec.Buffer],
        indices_are_used_only_for_memory_accesses: bool,
        kernel_code: str,
        scalar_parameters: List[str],
        scalar_parameter_types: Dict[str, str],
        template: str) -> str:
    get_iteration_replacement       = '(blockDim.x * blockIdx.x + threadIdx.x)'
    get_iteration_group_replacement = 'blockIdx.x'
    if (not indices_are_used_only_for_memory_accesses): # or any_of_the_buffers_is_random_access:
        get_iteration_replacement       = '((blockDim.x * blockIdx.x + threadIdx.x) + _batch_offset_x * blockDim.x)'
        get_iteration_group_replacement = '(blockIdx.x + _batch_offset_x)'
    kernel_code = kernel_code.replace('GET_ITERATION()', get_iteration_replacement)
    kernel_code = kernel_code.replace('GET_ITERATION_WITHIN_BATCH()', 'threadIdx.x')
    kernel_code = kernel_code.replace('GET_BATCH_ID()', get_iteration_group_replacement)
    kernel_code = kernel_code.replace('GET_2D_ITERATION_WITHIN_BATCH_X', 'threadIdx.x')
    kernel_code = kernel_code.replace('GET_2D_ITERATION_WITHIN_BATCH_Y', 'threadIdx.y')
    kernel_code = kernel_code.replace('GET_2D_ITERATION_X', '(blockDim.x * (blockIdx.x + _batch_offset_x) + threadIdx.x)')
    kernel_code = kernel_code.replace('GET_2D_ITERATION_Y', '(blockDim.y * (blockIdx.y + _batch_offset_y) + threadIdx.y)')
    kernel_code = insert_offset_computations_for_buffers_that_contain_indices(buffers, kernel_code)
    kernel_code = insert_cuda_texture_readers(buffers, kernel_code)
    template = template.replace('//%CUDA_KERNEL_CODE%', kernel_code)
    parameters = ''
    for b in buffers:
        parameters += b.type + ' *' + b.buffer_name + ',\n'
    for sp in scalar_parameters:
        parameters += scalar_parameter_types[sp] + ' ' + sp + ',\n'
    parameters += 'size_t _batch_offset_x,\n'
    parameters += 'size_t _batch_offset_y,\n'
    return template.replace('//%CUDA_KERNEL_PARAMETER_DECLS%', parameters[:-2])


def insert_cuda_kernel_call(buffers: List[buffer_spec.Buffer], is_2d_kernel: bool,  is_reduction: bool, t: str) -> str:
    if is_reduction:
        kernel_call_code = 'const size_t num_blocks = 64;\n'\
                           'const size_t block_size = 256;\n'\
                           'const size_t smem_size = block_size * sizeof(%TYPE%);\n' \
                           'reduce<<<num_blocks, block_size, smem_size>>>(' \
                           '    h.cuda.%SOURCE_BUFFER_NAME%_d, h.cuda.intermediate_results_d, slice_sizes[0]);'
        kernel_call_code = kernel_call_code.replace('%SOURCE_BUFFER_NAME%', buffers[1].buffer_name)\
                                           .replace('%TYPE%', buffers[1].type)
    else:
        if is_2d_kernel:
            kernel_call_code = 'dim3 grid_size(slice_sizes[0], slice_sizes[1]);'\
                               'dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);'\
                               'cuda_kernel<<<grid_size, block_size>>>(' \
                               '    //%CUDA_KERNEL_PARAMETERS%);'
        else:
            kernel_call_code = 'cuda_kernel<<<slice_sizes[0], BLOCK_SIZE_X>>>(' \
                               '    //%CUDA_KERNEL_PARAMETERS%);'
    return t.replace('//%CUDA_KERNEL_CALL%', kernel_call_code)


def insert_cuda_kernel_parameters(
        buffers: List[buffer_spec.Buffer],
        indices_are_used_only_for_memory_accesses: bool,
        scalar_parameters: List[str],
        template: str):
    parameters = ''
    for b in buffers:
        parameters += 'h.cuda.' + b.buffer_name + '_d' + ',\n'
    for sp in scalar_parameters:
        parameters += sp + ',\n'
    parameters += 'offsets[0],\n'
    parameters += 'offsets[1],\n'
    return template.replace('//%CUDA_KERNEL_PARAMETERS%', parameters[:-2])


def insert_cuda_barrier(template: str):
    return template.replace('device_hopper::batch_barrier()', '__syncthreads()')


def insert_code_that_exits_the_manager_thread_if_the_slice_is_aborted(
        code: str,
        is_idempotent: bool,
        is_reduction: bool,
        preferred_device: Device):
    if not is_reduction:
        code = code.replace('//%REDUCTION_RETURN_IF_THE_SLICE_IS_ABORTED%\n', '')
    if (is_idempotent and preferred_device == Device.CPU) or \
       (is_reduction  and preferred_device == Device.CPU):
        code_that_returns_from_the_manager_thread = 'if (abort_slice) {\n'\
                                                        'timing::stop(timing::Type::GPU_KERNEL);\n'\
                                                        'return;\n'\
                                                    '}'
        code = code.replace('//%RETURN_IF_THE_SLICE_IS_ABORTED%', code_that_returns_from_the_manager_thread)
        if is_reduction:
            code = code.replace('//%RETURN_IF_THE_SLICE_IS_ABORTED_RIGHT_AFTER_THE_KERNEL_LAUNCH%', '')
        else:
            code = code.replace('//%RETURN_IF_THE_SLICE_IS_ABORTED_RIGHT_AFTER_THE_KERNEL_LAUNCH%', code_that_returns_from_the_manager_thread)

        code_that_checks_if_the_current_slice_should_be_killed = 'if (ctx.sched.kill_current_slice(h.device)) {\n' \
                                                                     'cleanup( & h);\n' \
                                                                     'ctx.sched.free_device(h.device);\n' \
                                                                     'abort_slice = true;\n' \
                                                                     'return;\n' \
                                                                 '}'
        if is_reduction:
            return_from_manager_thread_right_after_reduction_kernel_launch = 'if (abort_slice) {\n' \
                                                                             'transfer_random_access_buffers_from_device(intermediate_results,\n' \
                                                                             'in,\n' \
                                                                             'intermediate_buffer_size,\n' \
                                                                             'problem_size,\n' \
                                                                             'h);\n' \
                                                                             'cleanup(&h);\n' \
                                                                             'ctx.sched.free_device(h.device);\n' \
                                                                             'timing::stop(timing::Type::GPU_KERNEL);\n' \
                                                                             'return;\n' \
                                                                             '}'
            code = code.replace('//%REDUCTION_RETURN_IF_THE_SLICE_IS_ABORTED%\n',
                                return_from_manager_thread_right_after_reduction_kernel_launch)

            code_that_checks_if_the_current_slice_should_be_killed = 'if (ctx.sched.kill_current_slice(h.device)) {\n' \
                                                                         'abort_slice = true;\n' \
                                                                         'return;\n' \
                                                                     '}'
        code = code.replace('//%CHECK_IF_THE_SLICE_SHOULD_BE_KILLED%',
                            code_that_checks_if_the_current_slice_should_be_killed)
    else:
        code = code.replace('//%RETURN_IF_THE_SLICE_IS_ABORTED%', '') \
                   .replace('//%RETURN_IF_THE_SLICE_IS_ABORTED_RIGHT_AFTER_THE_KERNEL_LAUNCH%', '') \
                   .replace('//%CHECK_IF_THE_SLICE_SHOULD_BE_KILLED%', '')
    return code


def insert_cuda_local_memory_attribute(template: str):
    return template.replace('__attribute__((device_hopper_batch_shared))', '__shared__')


def insert_cuda_texture_readers(buffers: List[buffer_spec.Buffer], t: str) -> str:
    for b in buffers:
        if b.use_texture_on_gpu:
            s = e = t.find(b.buffer_name + '[')
            unmatched_braces = 0
            print(t[s - 10: s + 10])
            while t[e - 1] != ']' or unmatched_braces != 0:
                if t[e] == '[':
                    unmatched_braces += 1
                elif t[e] == ']':
                    unmatched_braces -= 1
                e += 1
            code_between_braces = t[s + len(b.buffer_name) + 1:e-1]
            t = t[:s] + b.buffer_name + '_tex_reader(' + code_between_braces + ')' + t[e:]
    return t


def insert_offset_computations_for_buffers_that_contain_indices(
        buffers: List[buffer_spec.Buffer],
        cuda_kernel_code: str) -> str:
    search_position = 0
    for b in buffers:
        if b.contains_indices:
            e = cuda_kernel_code[search_position:].find(b.buffer_name + '[')
            while e != -1:
                unmatched_braces = 0
                while cuda_kernel_code[e - 1] != ']' or unmatched_braces != 0:
                    if cuda_kernel_code[e] == '[':
                        unmatched_braces += 1
                    elif cuda_kernel_code[e] == ']':
                        unmatched_braces -= 1
                    e += 1
                cuda_kernel_code = cuda_kernel_code[:e] + ' - ' + b.buffer_name + '[0]' + cuda_kernel_code[e:]
                search_position = e + len(b.buffer_name) + 6 + search_position
                e = cuda_kernel_code[search_position:].find(b.buffer_name + '[')
                if e != -1:
                    e += search_position
    return cuda_kernel_code


def extract_structs(code: str) -> List[str]:
    structs = []
    last_struct_pos = -1
    while True:
        last_struct_pos = code.strip().find('typedef struct', last_struct_pos + 1)
        if last_struct_pos == -1:
             break
        s = e = last_struct_pos
        open_curly_braces = 0
        while code[e] != ';' or open_curly_braces != 0:
            if code[e] == '{':
                open_curly_braces += 1
            elif code[e] == '}':
                open_curly_braces -= 1
            e += 1
        structs.append(code[s:e + 1] + '\n')

    #s = e = code.find('#include')
    #code = code[:s] + '\n'.join(structs) + '\n' + code[e:]
    return structs, code


def extract_user_defined_functions_that_are_called_in_a_code_segment(code: str, kernel_code: str) -> Set[str]:
    ignored_tokens = ['GET_ITERATION', 'GET_BATCH_ID', 'GET_ITERATION_WITHIN_BATCH', '__attribute__', 'for', 'if', 'batch_barrier']

    # Extract function names from kernel code
    matcher = re.compile('\w*[\s]?(?=\()')
    called_functions = matcher.findall(kernel_code)
    called_functions = set(called_functions) # Remove duplicates
    if '' in called_functions:
        called_functions.remove('')
    if ' ' in called_functions:
        called_functions.remove(' ')
    for ign_t in ignored_tokens:
        if ign_t in called_functions:
            called_functions.remove(ign_t)

    # Construct list of functions recursively
    recursively_found_function_calls = set()
    for cf in called_functions:
        # Extract function bodies
        matcher = re.compile(r'#define.{}\([\w|, ]*\)[ ]*\\[\n|\r\n]((.*\\\n)*).*(\n)'.format(cf), re.MULTILINE)
        test = matcher.findall(code)
        if len(test) > 0:
            function_body = test[0][0]

            # Call this function recursively
            fs = extract_user_defined_functions_that_are_called_in_a_code_segment(code, function_body)
            recursively_found_function_calls.update(fs)
    called_functions.update(recursively_found_function_calls)

    # Filter parametrised macros
    parametrised_macros = []
    for f in called_functions:
        s = '#define.{}\('.format(f)
        matcher = re.compile('#define.{}\('.format(f))
        if matcher.search(code):
            parametrised_macros.append(f)
    for pm in parametrised_macros:
        called_functions.remove(pm)

    return set([f.strip() for f in called_functions])


def extract_macros_that_are_used_in_the_kernel(code: str) -> List[str]:
    ignored_macros = ['REPOSITORY_PATH', 'PREFERRED_DEVICE']

    # Extract parametrised macros
    matcher = re.compile(r'(?:#define)\s(?:\w)+\([\w|,\s]*\)\s(?:.*\\[\n|\r\n](?:(?:.*\\\n)*).*|.*)', re.MULTILINE)
    parametrised_macros = matcher.findall(code)

    # Extract non-parametrised macros
    matcher = re.compile(r'(?:#define)\s+(?:\w)+\s+.+',)
    non_parametrised_macros = matcher.findall(code)

    # Removed macros that should be ignored
    result = []
    for non_parametrised_macro in non_parametrised_macros:
        add_to_result = True
        for ignored_macro in ignored_macros:
            if non_parametrised_macro.find(ignored_macro) != -1:
                add_to_result = False
        if add_to_result:
            result.append(non_parametrised_macro)
    result += parametrised_macros

    return result


def extract_function_definitions(code: str, user_defined_functions_called_by_the_kernel: Set[str]) -> List[str]:
    function_definitions = []
    for function_name in user_defined_functions_called_by_the_kernel:
        matcher = re.compile(r'(?:inline\s)?(void|int|float|float2|double|T|T1|T2)\s{}(?:\s)?\('.format(function_name), re.MULTILINE)
        match = matcher.search(code)
        if match:
            s = e = match.start()
            open_curly_braces = 0
            while code[e-1] != '}' or open_curly_braces != 0:
                if code[e] == '{':
                    open_curly_braces += 1
                elif code[e] == '}':
                    open_curly_braces -= 1
                e += 1
            function_definitions.append(code[s:e])
    return function_definitions


def extract_is_idempotent(code: str) -> bool:
    for line in code.split('\n'):
        if line.find('opt_set_is_idempotent(true)') != -1:
            return True
    return False


def extract_preferred_device(code: str) -> Device:
    match = re.search('(?<=#define PREFERRED_DEVICE )\w{3}', code)
    if match is None:
        print('Error: Preferred device is not defined with "#define PREFERRED_DEVICE <DEVICE>"')
        sys.exit(1)
    device_str = match.group()
    if device_str == 'GPU':
        return Device.GPU
    elif device_str == 'CPU':
        return Device.CPU
    else:
        print('Error: Unknown device')
        sys.exit(1)


def extract_if_kernel_contains_get_iteration_group(kernel_code: str):
    return kernel_code.find('GET_BATCH_ID') != -1


def extract_any_of_the_buffers_is_random_access(buffers: List[buffer_spec.Buffer]):
    for b in buffers:
        if b.access_pattern == buffer_spec.access_pattern.RANDOM:
            return True
    return False


def extract_whether_the_parallel_for_implements_a_reduction(code: str) -> bool:
    m = re.search(r'set_is_reduction\(', code)
    return not m is None


def extract_the_reduction_operator(code: str) -> reduction.Operator:
    # Extract the operator
    m = re.search(r'(?<=device_hopper::reduction_operation::)\w{3}(?=\w*|,)', code)
    if m is None:
        print('Error: The reduction operator has not been specified.')
        sys.exit(1)
    operator_str = m.group()
    if operator_str == 'ADD':
        op = reduction.Operator.ADD
    else:
        print('Error: The specified reduction operator is not supported yet.')
    return op


def extract_where_the_result_of_the_reduction_will_be_stored(
        code: str,
        scalar_parameters: List[str],
        scalar_parameter_types: Dict[str, str]):
    # Extract the variable that will contain the result
    m = re.search(r'(?<=device_hopper::reduction_operation::\w{3}(\s|,)[\s*])\w*(?=\);)', code)
    if m is None:
        print('Error: A variable that will contain the result of the reduction has not been set.')
        sys.exit(1)
    reduction_dest = m.group()

    # Extract the type of the destination variable
    m = re.search(r'(?<=\s)\w*(?=\sresult)', code)
    if m is None:
        print('Error: Could not extract the type of the destination variable of the reduction')
        sys.exit(1)
    reduction_dest_type = m.group() + '&'

    scalar_parameters.append(reduction_dest)
    scalar_parameter_types[reduction_dest] = reduction_dest_type


def insert_opencl_buffer_creation(buffers: List[buffer_spec.Buffer], template: str) -> str:
    buffer_creation_template = 'h.opencl.%DEVICE_HANDLE% = clCreateBuffer(\n'\
                                   'h.ctx,\n'\
                                   'CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,\n'\
                                   '%SIZE_IN_BYTES%,\n'\
                                   '%HOST_POINTER%,\n'\
                                   '&err);\n'\
                               'plasticity::cl_utils::cl_check_return_code(err, "Could not create an OpenCL buffer");\n'

    buffer_creation_code = ''
    for b in buffers:
        buffer_creation_code += buffer_creation_template.replace('%DEVICE_HANDLE%', b.buffer_name + '_d')\
                                 .replace('%SIZE_IN_BYTES%', b.element_count + ' * ' + 'sizeof(' + b.type + ')')\
                                 .replace('%HOST_POINTER%', b.buffer_name)
    return template.replace('//%OPENCL_BUFFER_CREATION%', buffer_creation_code)


def insert_opencl_kernel_parameters(
        buffers: List[buffer_spec.Buffer],
        kernel_code_contains_get_iteration_group: bool,
        indices_are_used_only_for_memory_accesses: bool,
        preferred_device: Device,
        scalar_parameters: List[str],
        scalar_parameter_types: Dict[str, str],
        template: str):
    parameter_template = 'err |= clSetKernelArg(h.kernel, %PARAMETER_COUNT%, sizeof(%PARAMETER_TYPE%), &%PARAMETER%);\n'

    parameter_code = ''
    parameter_count = 0
    for b in buffers:
        parameter_code += parameter_template.replace('%PARAMETER_COUNT%', str(parameter_count))\
                                            .replace('%PARAMETER_TYPE%', 'cl_mem')\
                                            .replace('%PARAMETER%', 'h.opencl.' + b.buffer_name + '_d')
        parameter_count += 1
    for sp in scalar_parameters:
        parameter_code += parameter_template.replace('%PARAMETER_COUNT%', str(parameter_count)) \
                                            .replace('%PARAMETER_TYPE%', scalar_parameter_types[sp]) \
                                            .replace('%PARAMETER%', sp)
        parameter_count += 1
    #if preferred_device != Device.CPU:
    if kernel_code_contains_get_iteration_group:
        code_for_offset_parameter = parameter_template.replace('%PARAMETER_COUNT%', str(parameter_count))\
                                                      .replace('%PARAMETER_TYPE%', 'int')\
                                                      .replace('%PARAMETER%', 'offsets[0]')
        template = template.replace('//%DYNAMIC_OPENCL_KERNEL_PARAMETERS%', code_for_offset_parameter)
    #else:
    #   template = template.replace('//%DYNAMIC_OPENCL_KERNEL_PARAMETERS%', '')
    return template.replace('//%OPENCL_KERNEL_ARGUMENTS%', parameter_code)


def insert_code_that_parses_cli_parameters(code):
    variables_map_instanciation = 'boost::program_options::variables_map vm;'
    s = e = code.find(variables_map_instanciation)
    code = code[:s] + 'plasticity::setup::Context context(argc, argv, desc);\n' + code[e:]
    code = code.replace(variables_map_instanciation, 'boost::program_options::variables_map vm = context.get_variables_map();')
    code = code.replace('boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);', '')
    return code


def insert_cuda_device_qualifiers_to_function_definitions(code, function_names):
    for fn in function_names:
        matcher = re.compile(r'(inline\s)?(void|int|float|float2|double|T|T1|T2)\s{}(\s)?\('.format(fn))
        matches = matcher.search(code)
        if matches:
            s = matches.start()
            code = code[:s] + '__device__ __host__ ' + code[s:]
    return code


def insert_the_kernel_into_the_opencl_template(
        indices_are_used_only_for_memory_accesses: bool,
        kernel: str,
        opencl_template: str,
        preferred_device: Device) -> str:
    get_iteration_replacement = 'get_global_id(0)'
    #if not indices_are_used_only_for_memory_accesses:
    #    get_iteration_replacement = ' + grain_count_offset_x * BLOCK_SIZE'
    kernel = kernel.replace('GET_ITERATION()', get_iteration_replacement)
    kernel = kernel.replace('GET_ITERATION_WITHIN_BATCH()', 'get_local_id(0)')
    kernel = kernel.replace('GET_BATCH_ID()', '(get_group_id(0) + _batch_offset)')
    kernel = kernel.replace('GET_2D_ITERATION_X', 'get_global_id(0)')
    kernel = kernel.replace('GET_2D_ITERATION_Y', 'get_global_id(1)')
    kernel = kernel.replace('GET_2D_ITERATION_WITHIN_BATCH_X', 'get_local_id(0)')
    kernel = kernel.replace('GET_2D_ITERATION_WITHIN_BATCH_Y', 'get_local_id(1)')

    return opencl_template.replace('%KERNEL_BODY%', kernel)


def insert_early_abort_related_parameters(template: str, preferred_device: Device):
    if preferred_device == Device.GPU:
        template = template\
            .replace('//%ABORT_SLICE_FLAG%', '') \
            .replace('//%ABORT_SLICE_FLAG_DECL%', '')
    else:
        template = template\
            .replace('//%ABORT_SLICE_FLAG%', 'abort_slice,') \
            .replace('//%ABORT_SLICE_FLAG_DECL%', 'bool& abort_slice,')
    return template


def insert_opencl_kernel_parameters_into_the_opencl_file(
        buffers: List[buffer_spec.Buffer],
        kernel_code_contains_get_iteration_group: bool,
        indices_are_used_only_for_memory_accesses: bool,
        preferred_device: Device,
        scalar_parameters: List[str],
        scalar_parameter_types: Dict[str, str],
        opencl_template: str) -> str:
    parameters = ''
    for b in buffers:
        parameters += '__global '
        if b.access_direction == buffer_spec.access_direction.IN:
            parameters += 'const '
        parameters += b.type + ' *' + b.buffer_name + ',\n'
    for sp in scalar_parameters:
        parameters += scalar_parameter_types[sp] + ' ' + sp + ',\n'
    # if not indices_are_used_only_for_memory_accesses:
    #     parameters += 'size_t grain_count_offset_x'
    #if preferred_device != Device.CPU:
    if kernel_code_contains_get_iteration_group:
        parameters += 'const int _batch_offset,\n'
    return opencl_template.replace('%KERNEL_PARAMETERS%', parameters[:-2])


def insert_structs_into_the_opencl_file(opencl_template: str, structs: List[str]):
    return '\n'.join(structs) + '\n' + opencl_template


def insert_address_space_qualifiers(buffers: List[buffer_spec.Buffer], opencl_t: str):
    for b in buffers:
        split_code = opencl_t.split('\n')
        line_count = -1
        for line in split_code:
            line_count += 1
            if b.buffer_name in line and '=' in line:
                match_buffer_access = re.search('{}\[.*\]'.format(b.buffer_name), line)
                match_reference_to_buffer_element = re.search('&{}\[.*\]'.format(b.buffer_name), line)
                if not match_buffer_access or match_reference_to_buffer_element:
                    split_line = line.strip().split(' ')
                    if split_line[0].strip() == 'struct':
                        split_line[0] = split_line[0].strip() + ' ' + split_line[1].strip()
                        del split_line[1]
                    if split_line[0] == b.type:
                        s = line.find(b.type)
                        line = line[:s] + '__global ' + line[s:]
                        split_code[line_count] = line

            # Process casts
            if line_count == 180:
                pass
                #print('here')
            matcher = re.compile(r'\w[\s|\w|*]*=.+\((?:\w|\s|\*)*\)\s+{}'.format(b.buffer_name))
            matches = matcher.findall(line)
            for m in matches:
                # Add address space qualifier to the type of the new pointer
                s = line.find(m)
                line = line[:s] + '__global ' + line[s:]
                # Add address space qualifier to the cast
                match = re.search('(?<=\()\w*\s\*\)', line)
                s = match.start()
                line = line[:s] + '__global ' + line[s:]
                split_code[line_count] = line
                # Create a new buffer object
                new_buffer = copy.deepcopy(b)
                split_cast_line = [c.replace('*', '').strip() for c in line.strip().split(' ')]
                new_buffer.buffer_name = split_cast_line[2]
                new_buffer.type        = split_cast_line[1]
                buffers.append(new_buffer)
        opencl_t = '\n'.join(split_code)

    # Process user defined functions that are called from within the kernel
    split_code = opencl_t.split('\n')
    line_count = -1
    for line in split_code:
        line_count += 1
        for b in buffers:
            matches = re.finditer(r'(?<=,).*(?:{} \*{}).*(?=,)'.format(b.type, b.buffer_name), line)
            for match in matches:
                # Check if the matched string already contains an address space qualifier
                s = match.start()
                e = match.end()
                if line[s:e].find(b.opencl_address_space_qualifier()) == -1:
                    line = line[:s] + b.opencl_address_space_qualifier() + line[s:]
                    split_code[line_count] = line
    opencl_t = '\n'.join(split_code)

    return opencl_t


def insert_opencl_file_name(opencl_file_name: str, opencl_template: str):
    return opencl_template.replace('%OPENCL_FILE_NAME_PREFIX%', opencl_file_name[:-3])


def insert_method_calls_that_set_processed_slice_size_and_offset(is_idempotent: bool, opencl_t: str) -> str:
    if is_idempotent:
        book_keeping_code = 'ctx.sched.set_old_offset_and_slice_size(current_offsets, current_slice_sizes);\n'
        opencl_t = opencl_t.replace('//%SET_OLD_OFFSET_AND_SLICE_SIZE_FOR_IDEMPOTENT_KERNEL%', book_keeping_code)\
                           .replace('//%SET_OLD_OFFSET_AND_SLICE_SIZE_FOR_NON_IDEMPOTENT_KERNEL%', '')
    else:
        code_for_non_idempotent_kernels = 'ctx.sched.book_keeping.lock();\n'\
                                          'if (ctx.sched.kill_current_slice(h.device)) {\n'\
                                               'if (dev == Device::GPU) transfer_random_access_buffers_from_device(\n'\
                                                    '//%BUFFERS%\n'\
                                                    '//%BUFFER_ELEMENT_COUNTS%\n'\
                                                    '//%BUFFER_ELEMENT_SIZES%\n'\
                                                    'h);\n'\
                                               'cleanup(&h);\n'\
                                               'ctx.sched.free_device(h.device);\n'\
                                               'timing::stop(dev == Device::CPU ? timing::Type::CPU_KERNEL : timing::Type::GPU_KERNEL);\n'\
                                               'ctx.sched.book_keeping.unlock();\n'\
                                               'return;\n'\
                                          '}\n'\
                                          'ctx.sched.set_old_offset_and_slice_size(current_offsets, current_slice_sizes);\n'\
                                          'ctx.sched.book_keeping.unlock();\n'
        opencl_t = opencl_t.replace('//%SET_OLD_OFFSET_AND_SLICE_SIZE_FOR_NON_IDEMPOTENT_KERNEL%',
                                    code_for_non_idempotent_kernels)\
                           .replace('//%SET_OLD_OFFSET_AND_SLICE_SIZE_FOR_IDEMPOTENT_KERNEL%', '')
    return opencl_t


def insert_opencl_barrier(opencl_template: str) -> str:
    return opencl_template.replace('device_hopper::batch_barrier()', 'barrier(CLK_LOCAL_MEM_FENCE)')


def insert_opencl_local_memory_attribute(opencl_template: str, buffers: List[buffer_spec.Buffer]) -> str:
    # Insert attribute
    opencl_template = opencl_template.replace('__attribute__((device_hopper_batch_shared))', '__local')

    # Add buffer to the list of buffers
    matches = re.finditer(r'(?<=__local\s)(?:\w|\s)*(?=;|\[)', opencl_template)
    for match in matches:
        s = match.start()
        e = match.end()
        partial_decl = opencl_template[s:e]
        type, name = partial_decl.split(' ')
        b = buffer_spec.Buffer(
            buffer_spec.access_pattern.RANDOM,
            buffer_spec.access_direction.INOUT,
            buffer_spec.address_space.SHARED_WITHIN_GRAIN,
            name,
            '',
            type
        )
        buffers.append(b)

    return opencl_template


def insert_function_definitions(function_definitions: List[str], opencl_t: str):
    for function_definition in function_definitions:
        opencl_t = function_definition + '\n\n' + opencl_t
    return opencl_t


def insert_macros(macros, opencl_t):
    for macro in macros:
        opencl_t = macro + '\n\n' + opencl_t
    return opencl_t


def insert_batch_sizes(block_size: Tuple[int, int], opencl_t: str) -> str:
    opencl_t = '#define BLOCK_SIZE_X ' + str(block_size[0]) + '\n' + \
               '#define BLOCK_SIZE_Y ' + str(block_size[1]) + '\n' + opencl_t
    return opencl_t


def insert_call_that_tells_the_scheduler_that_the_kernel_is_idempotent(is_idempotent: bool, t: str) -> str:
    if is_idempotent:
        t = t.replace('//%SET_IF_KERNEL_IS_IDEMPOTENT%', 'ctx.sched.set_kernel_is_idempotent();')
    return t


def insert_openmp_reduction(
        buffers: List[buffer_spec.Buffer],
        reduction_operator: reduction.Operator,
        scalar_parameters: List[str],
        t: str) -> str:
    # Load the template for the OpenMP reduction
    f = open('openmp_reduction_template.h', 'r')
    reduction_template = f.read()
    f.close()

    # Fill in the placeholders
    if len(buffers) > 2:
        print('Error: Reductions with more than one input buffer are not supported yet.')
        sys.exit(1)
    if len(scalar_parameters) > 1:
        print('Error: Only one variable for the fnal result is allowed.')
        sys.exit(1)
    reduction_template = reduction_template.replace('%REDUCTION_DESTINATION%', scalar_parameters[0]) \
                          .replace('%REDUCTION_OPERATOR%', reduction.operator_to_str(reduction_operator)) \
                          .replace('%INPUT_BUFFER_NAME%', str(buffers[1].buffer_name))

    # Paste the reduction code into the header file
    t = t.replace('//%OPENMP_KERNEL_CODE%', reduction_template)

    return t


def insert_a_buffer_for_intermediate_results_of_the_gpu_reduction_kernel(
        code: str,
        buffers: List[buffer_spec.Buffer],
        scalar_parameters: List[str],
        scalar_paramter_types: Dict[str, str]) -> str:
    buffer_allocation_code = 'size_t intermediate_buffer_size = 64;\n'\
                             '%TYPE% *intermediate_results  = (%TYPE% *) device_hopper::malloc(64, sizeof(%TYPE%));\n'\
                             'for (size_t i = 0; i < 64; ++i) intermediate_results[i] = 0;'
    # Check if the list and dictionary are populated as expected
    if len(scalar_parameters) > 1 or len(scalar_paramter_types.keys()) > 1:
        print('Error: Expected only one scalar paramter for reductions.')
        sys.exit(1)
    if len(scalar_parameters) < 1 or len(scalar_paramter_types.keys()) < 1:
        print('Error: Expected one scalar parameter. However, the respective data structures are empty.')
        sys.exit(1)

    # Insert the buffer allocation
    reduction_result_type  = scalar_paramter_types[scalar_parameters[0]].replace('&', '')
    buffer_allocation_code = buffer_allocation_code.replace('%TYPE%', reduction_result_type)
    s = code.find('parallel_for')
    if s == -1:
        print('Error: Could not find an instantiation of a parallel_for in the input code')
        sys.exit(1)

    # Add the buffer to the list of buffers that the translator uses internally
    b = buffer_spec.Buffer(
        access_pattern=buffer_spec.access_pattern.REDUCTION_INTERMEDIATE_RESULTS,
        access_direction=buffer_spec.access_direction.INOUT,
        address_space=buffer_spec.address_space.GLOBAL,
        buffer_name='intermediate_results',
        element_count='intermediate_buffer_size',
        type='double'
    )
    buffers.append(b)

    return code[:s] + buffer_allocation_code + code[s:]


def insert_cuda_kernel_stub(t: str):
    cuda_kernel_stub = ' __global__ void cuda_kernel('\
                                '//%CUDA_KERNEL_PARAMETER_DECLS%'\
                                ') {'\
                            '//%CUDA_KERNEL_CODE%'\
                        '}'
    return t.replace('//%CUDA_KERNEL%', cuda_kernel_stub)


def insert_cuda_reduction_kernel(
        reduction_operator: reduction.Operator,
        scalar_parameters: List[str],
        scalar_parameter_types: Dict[str, str],
        t: str) -> str:
    reduction_template_file = open('./cuda_reduction_template.h', 'r')
    reduction_template = reduction_template_file.read()
    reduction_template_file.close()
    reduction_result_type = scalar_parameter_types[scalar_parameters[0]].replace('&', '')
    reduction_template = reduction_template.replace('%TYPE%', reduction_result_type)
    return t.replace('//%CUDA_KERNEL%', reduction_template)


def insert_compute_final_reduction_results(
        scalar_parameters: List[str],
        t: str):
    code = 'for (size_t i = 0; i < 64; ++i) %REDUCTION_DESTINATION% += intermediate_results[i];'
    code = code.replace('%REDUCTION_DESTINATION%', scalar_parameters[0])
    return t.replace('//%COMPUTE_FINAL_REDUCTION_RESULT_BASED_ON_THE_INTERMEDIATE_RESULTS%', code)


def compile_opencl_kernel(compiler_flags: str, path_to_opencl_file: str):
    path_to_opencl_files = os.environ['PLASTICITY_ROOT'] + '/benchmarks/opencl_files/'
    args = [os.environ['PLASTICITY_ROOT'] + '/cmake-build-debug/utils/ocl_binary_generator',
            '--kernel-name=opencl_kernel',
            '--cl-file=' + path_to_opencl_files,
            '--cl-file-name=' + path_to_opencl_file.split('/')[-1],
            '--output-directory=' + path_to_opencl_files,
            '--target-device=CPU']
    if compiler_flags != None and compiler_flags != '':
        print(compiler_flags)
        args.append('--cl-compiler-flags="' + compiler_flags + '"')
    print(''.join(args))
    p = subprocess.Popen(args, stdout=subprocess.PIPE)
    p.wait()
    out = p.stdout.read()
    print(out.decode('utf8'))


def format_source_code(path_to_file: str):
    args = ('clang-format',
            '-i',
            path_to_file,
            '--style=Mozilla')
    p = subprocess.Popen(args, stdout=subprocess.PIPE)
    p.wait()
    out = p.stdout.read()
    if out != b'':
        print(out)


def main():
    # Parse CLI parameters
    p = argparse.ArgumentParser()
    p.add_argument('--input')
    p.add_argument('--output')
    p.add_argument('--opencl-compiler-flags', default='')
    args = p.parse_args()

    if args.input is None or args.output is None:
        print('Set --input or --output')
        exit(1)

    if not is_cuda_file(args.output):
        print('The output file has to be a CUDA file that ends with .cu')
    header_file_name      = args.output[0:-3] + '.h'
    path_to_opencl_file   = args.output[0:-3] + '.cl'
    opencl_compiler_flags = args.opencl_compiler_flags
    path_components       = path_to_opencl_file.split('/')
    path_components.insert(-1, 'opencl_files')
    path_to_opencl_file   = '/'.join(path_components)
    opencl_file_name      = path_to_opencl_file.split('/')[-1].strip()

    f = open(args.input, 'r')
    code = f.read()
    f.close()

    # Check if the computation is a reduction
    is_reduction           = extract_whether_the_parallel_for_implements_a_reduction(code)

    # Extract information from the input file
    buffers = []
    scalar_parameters      = extract_scalar_parameters(code)
    scalar_parameter_types = extract_scalar_parameter_types(code, scalar_parameters)
    if is_reduction:
        reduction_operator = extract_the_reduction_operator(code)
        extract_where_the_result_of_the_reduction_will_be_stored(code, scalar_parameters, scalar_parameter_types)
        # Insert an extra buffer for intermediate results for the GPU kernel in case the computation is a reduction
        code = insert_a_buffer_for_intermediate_results_of_the_gpu_reduction_kernel(
                code, buffers, scalar_parameters, scalar_parameter_types)
    # Extract further information from the source code
    is_2d_kernel = extract_whether_the_kernel_has_2d_iteration_space(code)
    batch_size_is_set = extract_if_batch_size_is_set(code, is_reduction)
    if batch_size_is_set:
        batch_sizes = extract_batch_size(code, is_2d_kernel, is_reduction)
        code = code.replace('pf.batch_size', str(batch_sizes[0]))
    extract_buffer_info(buffers, code)
    extract_indirect_accesses(buffers)
    indices_are_used_only_for_memory_accesses = extract_if_indices_are_used_only_for_memory_accesses(code)
    iteration_space_starts, iteration_space_ends = extract_iteration_space_bounds(code, is_2d_kernel)
    kernel_code                                = extract_kernel_code(code)
    structs, code                              = extract_structs(code)
    names_of_user_defined_functions_called_by_the_kernel = extract_user_defined_functions_that_are_called_in_a_code_segment(code, kernel_code)
    macros                                     = extract_macros_that_are_used_in_the_kernel(code)
    function_definitions                       = extract_function_definitions(code, names_of_user_defined_functions_called_by_the_kernel)
    is_idempotent                              = extract_is_idempotent(code)
    preferred_device                           = extract_preferred_device(code)
    kernel_code_contains_get_iteration_group   = extract_if_kernel_contains_get_iteration_group(kernel_code)
    any_of_the_buffers_is_random_access        = extract_any_of_the_buffers_is_random_access(buffers)

    # Generate the .cu file that contains the main function
    if batch_size_is_set:
        code = insert_the_block_size(batch_sizes, batch_size_is_set, code)
    code = insert_misc_defines(buffers, code, is_2d_kernel, is_idempotent, is_reduction, preferred_device)
    code = insert_header_file(header_file_name, code)
    code = insert_parallel_for(buffers, code, is_reduction, iteration_space_starts[0], iteration_space_starts[1],
                               iteration_space_ends[0], iteration_space_ends[1], scalar_parameters)
    code = insert_code_that_parses_cli_parameters(code)
    code = insert_cuda_device_qualifiers_to_function_definitions(code, names_of_user_defined_functions_called_by_the_kernel)
    code = insert_cuda_barrier(code)

    # Read template
    t = read_the_template()

    # Generate header file for the parallel_for based on the template
    t = insert_boiler_plate_code_for_textures(buffers, t)
    t = insert_parameters_for_indirect_buffer_accesses(buffers, t)
    t = insert_scalar_parameter_declarations(buffers, scalar_parameters, scalar_parameter_types, t)
    t = insert_opencl_buffer_handles(buffers, t)
    t = insert_cuda_buffer_handles(buffers, t)
    t = insert_cuda_buffer_creation(indices_are_used_only_for_memory_accesses, buffers, t)
    t = insert_cleanup_code(buffers, t)
    t = insert_code_for_data_transfers_to_the_gpu(indices_are_used_only_for_memory_accesses, is_reduction, buffers, preferred_device, t)
    t = insert_code_for_data_transfers_from_the_gpu(indices_are_used_only_for_memory_accesses, is_reduction, buffers, is_idempotent, preferred_device, t)
    t = insert_buffer_related_function_parameter_declarations(buffers, t)
    t = insert_early_abort_related_parameters(t, preferred_device)
    t = insert_cuda_kernel_call(buffers, is_2d_kernel, is_reduction, t)
    if is_reduction:
        t = insert_cuda_reduction_kernel(reduction_operator, scalar_parameters, scalar_parameter_types, t)
        t = insert_compute_final_reduction_results(scalar_parameters, t)
    else:
        t = insert_cuda_kernel_stub(t)
        t = insert_texture_object_instantiations(buffers, t)
        t = insert_cuda_kernel(any_of_the_buffers_is_random_access, buffers, indices_are_used_only_for_memory_accesses,
                               kernel_code, scalar_parameters, scalar_parameter_types, t)
        t = insert_cuda_kernel_parameters(buffers, indices_are_used_only_for_memory_accesses, scalar_parameters, t)
        t = insert_cuda_local_memory_attribute(t)


    t = insert_opencl_buffer_creation(buffers, t)
    t = insert_opencl_kernel_parameters(buffers, kernel_code_contains_get_iteration_group,
                                        indices_are_used_only_for_memory_accesses, preferred_device,
                                        scalar_parameters, scalar_parameter_types, t)
    t = insert_opencl_file_name(opencl_file_name, t)
    t = insert_method_calls_that_set_processed_slice_size_and_offset(is_idempotent, t)
    t = insert_code_that_exits_the_manager_thread_if_the_slice_is_aborted(t, is_idempotent, is_reduction, preferred_device)
    t = insert_call_that_tells_the_scheduler_that_the_kernel_is_idempotent(is_idempotent, t)
    if is_reduction:
        t = insert_openmp_reduction(buffers, reduction_operator, scalar_parameters, t)
    t = insert_cuda_barrier(t)
    t = insert_buffer_related_function_parameters(buffers, t)
    t = insert_scalar_parameters(buffers, scalar_parameters, scalar_parameter_types, t)

    # Generate OpenCL kernel
    opencl_t = read_the_opencl_template()
    opencl_t = insert_the_kernel_into_the_opencl_template(indices_are_used_only_for_memory_accesses, kernel_code,
                                                          opencl_t, preferred_device)
    opencl_t = insert_opencl_kernel_parameters_into_the_opencl_file(buffers, kernel_code_contains_get_iteration_group,
                                                                    indices_are_used_only_for_memory_accesses,
                                                                    preferred_device, scalar_parameters,
                                                                    scalar_parameter_types, opencl_t)
    opencl_t = insert_opencl_local_memory_attribute(opencl_t, buffers)
    if batch_size_is_set:
        opencl_t = insert_batch_sizes(batch_sizes, opencl_t)
    opencl_t = insert_function_definitions(function_definitions, opencl_t)
    opencl_t = insert_structs_into_the_opencl_file(opencl_t, structs)
    opencl_t = insert_macros(macros, opencl_t)
    opencl_t = insert_opencl_barrier(opencl_t)
    opencl_t = insert_address_space_qualifiers(buffers, opencl_t)

    # Write OpenCL file to disk
    if not is_reduction:
        f = open(path_to_opencl_file, 'w')
        f.write(opencl_t)
        f.close()

    # Write header file to disk
    f = open(header_file_name, 'w')
    f.write(t)
    f.close()

    # Write translated program code to disk
    f = open(args.output, 'w')
    f.write(code)
    f.close()

    # Format header file
    format_source_code(header_file_name)

    # Format translated program
    format_source_code(args.output)

    # Precompile the OpenCL code
    if not is_reduction:
        compile_opencl_kernel(opencl_compiler_flags, path_to_opencl_file)

    print('Done')

main()
