







#include <cuda_runtime.h>




struct upfirdn2d_kernel_params
{
    const void*     x;
    const float*    f;
    void*           y;

    int2            up;
    int2            down;
    int2            pad0;
    int             flip;
    float           gain;

    int4            inSize;         
    int4            inStride;
    int2            filterSize;     
    int2            filterStride;
    int4            outSize;        
    int4            outStride;
    int             sizeMinor;
    int             sizeMajor;

    int             loopMinor;
    int             loopMajor;
    int             loopX;
    int             launchMinor;
    int             launchMajor;
};




struct upfirdn2d_kernel_spec
{
    void*   kernel;
    int     tileOutW;
    int     tileOutH;
    int     loopMinor;
    int     loopX;
};




template <class T> upfirdn2d_kernel_spec choose_upfirdn2d_kernel(const upfirdn2d_kernel_params& p);


