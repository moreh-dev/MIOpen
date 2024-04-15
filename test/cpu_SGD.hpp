#ifndef GUARD_CPU_SGD_HPP
#define GUARD_CPU_SGD_HPP

#include "tensor_holder.hpp"

template <class T>
void cpu_SGD_forward(tensor<T> param_input,
                     tensor<T> ref_param_output,
                     tensor<T> grad,
                     tensor<T> momentum_buffer_input,
                     tensor<T> ref_momentum_buffer_output,
                     double lr,
                     double momentum,
                     double dampening,
                     double weight_decay,
                     char nesterov,
                     char momentum_initialized)
{
    auto dims = param_input.desc.GetLengths();
    size_t param_size = 0;
    for(size_t dim : dims)
    {
        param_size += dim;
    }

    for(int id = 0; id < param_size; ++id)
    {
        T param = param_input[id];
        T d_p = grad[id];

        if (weight_decay != 0)
        {
            d_p += param * weight_decay;
        }

        if (momentum != 0)
        {
            T momentum_v;
            if (momentum_initialized)
            {
                momentum_v = momentum_buffer_input[id];
                momentum_v = momentum_v * momentum + d_p * (1 - dampening);
            }
            else
            {
                momentum_v = d_p;
            }
            ref_momentum_buffer_output[id] = momentum_v;

            if (nesterov)
            {
                d_p = d_p + momentum_v * momentum;
            }
            else
            {
                d_p = momentum_v;
            }
        }

        ref_param_output[id] = param - lr * d_p;
    }
}
#endif
