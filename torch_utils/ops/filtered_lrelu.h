







#include <cuda_runtime.h>




struct filtered_lrelu_kernel_params
{
    
    int             up;         
    int             down;       
    int2            fuShape;    
    int2            fdShape;    

    int             _dummy;     

    
    const void*     x;          
    void*           y;          
    const void*     b;          
    unsigned char*  s;          
    const float*    fu;         
    const float*    fd;         

    int2            pad0;       
    float           gain;       
    float           slope;      
    float           clamp;      
    int             flip;       

    int             tilesXdim;  
    int             tilesXrep;  
    int             blockZofs;  

    int4            xShape;     
    int4            yShape;     
    int2            sShape;     
    int2            sOfs;       
    int             swLimit;    

    longlong4       xStride;    
    longlong4       yStride;    
    int64_t         bStride;    
    longlong3       fuStride;   
    longlong3       fdStride;   
};

struct filtered_lrelu_act_kernel_params
{
    void*           x;          
    unsigned char*  s;          

    float           gain;       
    float           slope;      
    float           clamp;      

    int4            xShape;     
    longlong4       xStride;    
    int2            sShape;     
    int2            sOfs;       
};




struct filtered_lrelu_kernel_spec
{
    void*   setup;              
    void*   exec;               
    int2    tileOut;            
    int     numWarps;           
    int     xrep;               
    int     dynamicSharedKB;    
};




template <class T, class index_t, bool signWrite, bool signRead> filtered_lrelu_kernel_spec choose_filtered_lrelu_kernel(const filtered_lrelu_kernel_params& p, int sharedKB);
template <class T, bool signWrite, bool signRead> void* choose_filtered_lrelu_act_kernel(void);
template <bool signWrite, bool signRead> cudaError_t copy_filters(cudaStream_t stream);


