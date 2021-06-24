// Taken from the SHOC benchmark suite

#ifndef PLASTICITY_FFTLIB_H
#define PLASTICITY_FFTLIB_H


#include <assert.h>
#include <cstdio>
#include <map>

#include "../../device_hopper/internal/utils.h"
#include "../../device_hopper/internal/cl_utils.h"


static std::map<void*, cl_mem> memobjmap;

extern const char *cl_source_fft;


void deinit(cl_command_queue fftQueue,
            cl_program& fftProg,
            cl_kernel& fftKrnl)
{
    for (std::map<void*, cl_mem>::iterator it = memobjmap.begin(); it != memobjmap.end(); ++it) {
        clEnqueueUnmapMemObject(fftQueue, it->second, it->first, 0, NULL, NULL);
        clReleaseMemObject(it->second);
    }

    clReleaseKernel(fftKrnl);
    clReleaseProgram(fftProg);
}

int transform(cl_mem workp,
              const size_t n_ffts,
              cl_kernel& fftKrnl,
              cl_command_queue& fftQueue) {
    cl_int err = 0;
    size_t localsz = 64u;
    size_t globalsz = localsz * n_ffts;

    err = clSetKernelArg(fftKrnl, 0, sizeof(cl_mem), &workp);
    err |= clEnqueueNDRangeKernel(fftQueue, fftKrnl, 1, NULL,
                                 &globalsz, &localsz, 0,
                                 NULL, NULL);
    return err;
}


cl_mem allocHostBuffer(void* bufp,
                     const unsigned long bytes,
                     cl_context fftCtx) {
    cl_int err;
    cl_mem memobj = clCreateBuffer(fftCtx,
                                   CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                   bytes,
                                   bufp,
                                   &err);
    plasticity::cl_utils::cl_check_return_code(err, "Could not create the host buffer");

    /**bufp = clEnqueueMapBuffer(fftQueue, memobj, true,
                               CL_MAP_READ | CL_MAP_WRITE,
                               0,bytes,0,NULL,NULL,&err);*/
    //memobjmap[*bufp] = memobj;
    //plasticity::cl_utils::cl_check_return_code(err, "Could not map the host buffer");
    return memobj;
}

int check(const void* workp,
          const void* checkp,
          const int half_n_ffts,
          const int half_n_cmplx,
          cl_kernel& chkKrnl,
          cl_command_queue& fftQueue)
{
    cl_int err = CL_SUCCESS;
    size_t localsz = 64;
    size_t globalsz = localsz * half_n_ffts;
    int result;

    clSetKernelArg(chkKrnl, 0, sizeof(cl_mem), workp);
    clSetKernelArg(chkKrnl, 1, sizeof(int), (void*)&half_n_cmplx);
    clSetKernelArg(chkKrnl, 2, sizeof(cl_mem), checkp);

    cl_event event;
    err |= clEnqueueNDRangeKernel(fftQueue, chkKrnl, 1, NULL,
                                 &globalsz, &localsz, 0,
                                 NULL, &event);
    err |= clFinish(fftQueue);
    err |= clEnqueueReadBuffer(fftQueue, *(cl_mem*)checkp, CL_TRUE, 0, sizeof(result),
                              &result, 1, &event, NULL);
    plasticity::cl_utils::cl_check_return_code(err, "FFT check failed");
    return result;
}

void freeHostBuffer(void* buf,
                    cl_context fftCtx,
                    cl_command_queue fftQueue)
{
    cl_int err;
    cl_mem memobj = memobjmap[buf];
    err = clEnqueueUnmapMemObject(fftQueue, memobj, buf, 0, NULL, NULL);
    plasticity::cl_utils::cl_check_return_code(err, "Could not unmap the host buffer");
    err = clReleaseMemObject(memobj);
    plasticity::cl_utils::cl_check_return_code(err, "Could not deallocate the host buffer");
    memobjmap.erase(buf);
}

void allocDeviceBuffer(void** bufferp,
                       const unsigned long bytes,
                       cl_context fftCtx,
                       cl_command_queue fftQueue)
{
    cl_int err;
    *(cl_mem**)bufferp = new cl_mem;
    **(cl_mem**)bufferp = clCreateBuffer(fftCtx, CL_MEM_READ_WRITE, bytes,
                                         NULL, &err);
    plasticity::cl_utils::cl_check_return_code(err, "Could not allocate the device  buffer");
}

void freeDeviceBuffer(void* buffer,
                      cl_context fftCtx,
                      cl_command_queue fftQueue)
{
    clReleaseMemObject(*(cl_mem*)buffer);
}

void copyToDevice(void* to_device, void* from_host,
                  const unsigned long bytes, cl_command_queue fftQueue)
{
    cl_int err = clEnqueueWriteBuffer(fftQueue, *(cl_mem*)to_device, CL_TRUE,
                                      0, bytes, from_host, 0, NULL, NULL);
    plasticity::cl_utils::cl_check_return_code(err, "Could not enqueue 'clEnqueueWriteBuffer'");
}

void copyFromDevice(void* to_host, void* from_device,
                    const unsigned long bytes, cl_command_queue fftQueue)
{
    cl_int err = clEnqueueReadBuffer(fftQueue, *(cl_mem*)from_device, CL_TRUE,
                                     0, bytes, to_host, 0, NULL, NULL);
    plasticity::cl_utils::cl_check_return_code(err, "Could not enqueue 'clEnqueueReadBuffer'");
}


#endif //PLASTICITY_FFTLIB_H
