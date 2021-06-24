texture<%TYPE%, 1> %BUFFER_NAME%_tex;  // vector textures

struct %BUFFER_NAME%_tex_reader_struct {
    __device__ __forceinline__ float operator()(const int idx) const {
        // The IDE shows an error here but the CUDA compiler compiles this fine.
        return tex1Dfetch(%BUFFER_NAME%_tex, idx);
    }
};