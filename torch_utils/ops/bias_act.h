










struct bias_act_kernel_params
{
    const void* x;      
    const void* b;      
    const void* xref;   
    const void* yref;   
    const void* dy;     
    void*       y;      

    int         grad;
    int         act;
    float       alpha;
    float       gain;
    float       clamp;

    int         sizeX;
    int         sizeB;
    int         stepB;
    int         loopX;
};




template <class T> void* choose_bias_act_kernel(const bias_act_kernel_params& p);


