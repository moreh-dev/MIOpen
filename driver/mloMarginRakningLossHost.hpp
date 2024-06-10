/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#ifndef MLO_MARGINRANKINGLOSS_MHOST_H_
#define MLO_MARGINRANKINGLOSS_MHOST_H_

#include <miopen/tensor.hpp>
#include <miopen/tensor_view.hpp>

template <typename Tgpu, typename Tcheck>
int32_t mloMarginRankingLossReducedForwardRunHost(const miopenTensorDescriptor_t input1Desc,
                                                  const Tgpu* input1,
                                                  const miopenTensorDescriptor_t input2Desc,
                                                  const Tgpu* input2,
                                                  const miopenTensorDescriptor_t targetDesc,
                                                  const Tgpu* target,
                                                  const miopenTensorDescriptor_t outputDesc,
                                                  Tcheck* output,
                                                  float margin,
                                                  float divisor)
{
    tensor_view_5d_t I1_tv = get_inner_expanded_tv_5d(miopen::deref(input1Desc));
    tensor_view_5d_t I2_tv = get_inner_expanded_tv_5d(miopen::deref(input2Desc));
    tensor_view_5d_t T_tv = get_inner_expanded_tv_5d(miopen::deref(targetDesc));
    tensor_view_5d_t O_tv = get_inner_expanded_tv_5d(miopen::deref(outputDesc));
    size_t tensor_size = miopen::deref(targetDesc).GetElementSize();
    size_t n[5];

    for(size_t idx = 0; idx < tensor_size; ++idx)
    {
        GET_NCDHW(n[0], n[1], n[2], n[3], n[4], idx, T_tv)
        size_t I1idx = TV5D_IDX(I1_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t I2idx = TV5D_IDX(I2_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t Tidx = TV5D_IDX(T_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t Oidx = TV5D_IDX(O_tv, n[0], n[1], n[2], n[3], n[4]);

        output[Oidx] = - target[Tidx] * (input1[I1idx] - input2[I2idx]) + static_cast<Tgpu>(margin);
        if (output[Oidx] < 0) output[Oidx] = 0.0f;
        output[Oidx] /= divisor;
    }

    return 0;
}

template <typename Tgpu, typename Tcheck>
int32_t mloMarginRankingLossReducedBackwardRunHost(const miopenTensorDescriptor_t input1Desc,
                                                   const Tgpu* input1,
                                                   const miopenTensorDescriptor_t input2Desc,
                                                   const Tgpu* input2,
                                                   const miopenTensorDescriptor_t targetDesc,
                                                   const Tgpu* target,
                                                   const miopenTensorDescriptor_t outGradDesc,
                                                   const Tgpu* outGrad,
                                                   const miopenTensorDescriptor_t in1GradDesc,
                                                   Tcheck* in1Grad,
                                                   const miopenTensorDescriptor_t in2GradDesc,
                                                   Tcheck* in2Grad,
                                                   float margin,
                                                   float divisor)
{
    tensor_view_5d_t I1_tv = get_inner_expanded_tv_5d(miopen::deref(input1Desc));
    tensor_view_5d_t I2_tv = get_inner_expanded_tv_5d(miopen::deref(input2Desc));
    tensor_view_5d_t T_tv = get_inner_expanded_tv_5d(miopen::deref(targetDesc));
    tensor_view_5d_t dO_tv = get_inner_expanded_tv_5d(miopen::deref(outGradDesc));
    tensor_view_5d_t dI1_tv = get_inner_expanded_tv_5d(miopen::deref(in1GradDesc));
    tensor_view_5d_t dI2_tv = get_inner_expanded_tv_5d(miopen::deref(in2GradDesc));
    size_t tensor_size = miopen::deref(targetDesc).GetElementSize();
    size_t n[5];

    for(size_t idx = 0; idx < tensor_size; ++idx)
    {
        GET_NCDHW(n[0], n[1], n[2], n[3], n[4], idx, T_tv)
        size_t I1idx = TV5D_IDX(I1_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t I2idx = TV5D_IDX(I2_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t Tidx = TV5D_IDX(T_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t dOidx = TV5D_IDX(dO_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t dI1idx = TV5D_IDX(dI1_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t dI2idx = TV5D_IDX(dI2_tv, n[0], n[1], n[2], n[3], n[4]);

        Tgpu t = - target[Tidx] * (input1[I1idx] - input2[I2idx]) + static_cast<Tgpu>(margin);

        if (t < 0) 
        {
            if (in1Grad) in1Grad[dI1idx] = 0.0f;
            if (in2Grad) in2Grad[dI2idx] = 0.0f;
        } else 
        {
            if (in1Grad) in1Grad[dI1idx] = - target[Tidx] * outGrad[dOidx] / divisor;
            if (in2Grad) in2Grad[dI2idx] = target[Tidx] * outGrad[dOidx] / divisor;
        }
    }

    return 0;
}

template <typename Tgpu, typename Tcheck>
int32_t mloMarginRankingLossUnreducedForwardRunHost(const miopenTensorDescriptor_t input1Desc,
                                                  const Tgpu* input1,
                                                  const miopenTensorDescriptor_t input2Desc,
                                                  const Tgpu* input2,
                                                  const miopenTensorDescriptor_t targetDesc,
                                                  const Tgpu* target,
                                                  const miopenTensorDescriptor_t outputDesc,
                                                  Tcheck* output,
                                                  float margin)
{
    tensor_view_5d_t I1_tv = get_inner_expanded_tv_5d(miopen::deref(input1Desc));
    tensor_view_5d_t I2_tv = get_inner_expanded_tv_5d(miopen::deref(input2Desc));
    tensor_view_5d_t T_tv = get_inner_expanded_tv_5d(miopen::deref(targetDesc));
    tensor_view_5d_t O_tv = get_inner_expanded_tv_5d(miopen::deref(outputDesc));
    size_t tensor_size = miopen::deref(targetDesc).GetElementSize();
    size_t n[5];

    for(size_t idx = 0; idx < tensor_size; ++idx)
    {
        GET_NCDHW(n[0], n[1], n[2], n[3], n[4], idx, T_tv)
        size_t I1idx = TV5D_IDX(I1_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t I2idx = TV5D_IDX(I2_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t Tidx = TV5D_IDX(T_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t Oidx = TV5D_IDX(O_tv, n[0], n[1], n[2], n[3], n[4]);

        output[Oidx] = - target[Tidx] * (input1[I1idx] - input2[I2idx]) + static_cast<Tgpu>(margin);
        if (output[Oidx] < 0) output[Oidx] = 0.0f;
    }

    return 0;
}

template <typename Tgpu, typename Tcheck>
int32_t mloMarginRankingLossUnreducedBackwardRunHost(const miopenTensorDescriptor_t input1Desc,
                                                  const Tgpu* input1,
                                                  const miopenTensorDescriptor_t input2Desc,
                                                  const Tgpu* input2,
                                                  const miopenTensorDescriptor_t targetDesc,
                                                  const Tgpu* target,
                                                  const miopenTensorDescriptor_t outGradDesc,
                                                  const Tgpu* outGrad,
                                                  const miopenTensorDescriptor_t in1GradDesc,
                                                  Tcheck* in1Grad,
                                                  const miopenTensorDescriptor_t in2GradDesc,
                                                  Tcheck* in2Grad,
                                                  float margin)
{
    tensor_view_5d_t I1_tv = get_inner_expanded_tv_5d(miopen::deref(input1Desc));
    tensor_view_5d_t I2_tv = get_inner_expanded_tv_5d(miopen::deref(input2Desc));
    tensor_view_5d_t T_tv = get_inner_expanded_tv_5d(miopen::deref(targetDesc));
    tensor_view_5d_t dO_tv = get_inner_expanded_tv_5d(miopen::deref(outGradDesc));
    tensor_view_5d_t dI1_tv = get_inner_expanded_tv_5d(miopen::deref(in1GradDesc));
    tensor_view_5d_t dI2_tv = get_inner_expanded_tv_5d(miopen::deref(in2GradDesc));
    size_t tensor_size = miopen::deref(targetDesc).GetElementSize();
    size_t n[5];

    for(size_t idx = 0; idx < tensor_size; ++idx)
    {
        GET_NCDHW(n[0], n[1], n[2], n[3], n[4], idx, T_tv)
        size_t I1idx = TV5D_IDX(I1_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t I2idx = TV5D_IDX(I2_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t Tidx = TV5D_IDX(T_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t dOidx = TV5D_IDX(dO_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t dI1idx = TV5D_IDX(dI1_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t dI2idx = TV5D_IDX(dI2_tv, n[0], n[1], n[2], n[3], n[4]);

        Tgpu t = - target[Tidx] * (input1[I1idx] - input2[I2idx]) + static_cast<Tgpu>(margin);

        if (t < 0) 
        {
            if (in1Grad) in1Grad[dI1idx] = 0.0f;
            if (in2Grad) in2Grad[dI2idx] = 0.0f;
        } else 
        {
            if (in1Grad) in1Grad[dI1idx] = - target[Tidx] * outGrad[dOidx];
            if (in2Grad) in2Grad[dI2idx] = target[Tidx] * outGrad[dOidx];
        }
    }

    return 0;
}

#endif // MLO_MARGINRANKINGLOSS_MHOST_H_
