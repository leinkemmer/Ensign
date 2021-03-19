#include <generic/utility.hpp>

#ifdef __CUDACC__
void* gpu_malloc(size_t size) {
    cout << "gpu_malloc: " << size/1e9 << " GB" << endl;
    void *p;
    if(cudaMalloc(&p, size) == cudaErrorMemoryAllocation) {
        cout << "ERROR: allocating " << size/1e9 << " GB on the gpu failed"
             << endl;
        exit(1);
    }
    return p;
}
#endif
