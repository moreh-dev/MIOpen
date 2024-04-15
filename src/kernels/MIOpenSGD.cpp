#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"

extern "C" __global__ void SGDFwdContiguous(const FLOAT* __restrict__ param_in,
                                            FLOAT* __restrict__ param_out,
                                            const FLOAT* __restrict__ grad,
                                            const FLOAT* __restrict__ momentum_buffer_in,
                                            FLOAT* __restrict__ momentum_buffer_out,
                                            double lr,
                                            double momentum,
                                            double dampening,
                                            double weight_decay,
                                            char nesterov,
                                            char momentum_initialized,
                                            size_t param_size)
{
    const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= param_size) return;

    FLOAT_ACCUM param = CVT_FLOAT2ACCUM(param_in[gid]);
    FLOAT_ACCUM d_p = CVT_FLOAT2ACCUM(grad[gid]);

    if (weight_decay != 0) 
    {
        d_p += param * CVT_FLOAT2ACCUM(weight_decay);
    }

    if (momentum != 0)
    {
        FLOAT_ACCUM momentum_v;
        if (momentum_initialized) 
        {
            momentum_v = CVT_FLOAT2ACCUM(momentum_buffer_in[gid]);
            momentum_v = momentum_v * CVT_FLOAT2ACCUM(momentum) + d_p * CVT_FLOAT2ACCUM(1 - dampening);
        }
        else 
        {
            momentum_v = d_p;
        }
        momentum_buffer_out[gid] = CVT_FLOAT2ACCUM(momentum_v);

        if (nesterov)
        {
            d_p = d_p + momentum_v * CVT_FLOAT2ACCUM(momentum);
        }
        else 
        {
            d_p = momentum_v;
        }
    }

    param_out[gid] = CVT_ACCUM2FLOAT(param - CVT_FLOAT2ACCUM(lr) * d_p);
}

                                            

