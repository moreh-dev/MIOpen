/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_TENSOR_REORDER_SOL_HPP
#define GUARD_MIOPEN_TENSOR_REORDER_SOL_HPP

#include <miopen/miopen.h>
#include <miopen/kernel_info.hpp>
#include <miopen/op_kernel_args.hpp>
#include <miopen/execution_context.hpp>
#include <vector>
#include <../kernels/gpu_general_tensor_reorder_kernel/order.hpp>

#include <miopen/tensor.hpp>
#include <miopen/magic_div.hpp>
#include <miopen/float_equal.hpp>
#include <string>
#include <limits>
#include <iostream>
#include <sstream>

#define TENSOR_REORDER_BLOCK_SIZE 256
#define TENSOR_REORDER_PERSISTENT 0

#if TENSOR_REORDER_PERSISTENT
#define TENSOR_REORDER_OCCUPANCY 4
#endif

namespace miopen {

struct GeneralReorderParam
{
    int tile_x{0};
    int tile_y{0};
    int pack_x{0};
    int pack_y{0};
    int ediv_x{0};
    int ediv_y{0};
};

template<typename dst_order>
struct GeneralReorderSolution
{
    GeneralReorderSolution(const ExecutionContext& ctx_,
                                miopenDataType_t data_type_,
                                uint32_t dim_0_,
                                uint32_t dim_1_,
                                uint32_t dim_2_,
                                uint32_t dim_3_);
    solver::KernelInfo GetKernel() const;
    std::vector<OpKernelArg> GetKernelArg() const;
    std::string GetKernelName() const;
    bool IsSkippable() const;
    size_t GetSize() const;

    miopenDataType_t data_type;
    uint32_t dim_0;
    uint32_t dim_1;
    uint32_t dim_2;
    uint32_t dim_3;
    int num_cu;

    GeneralReorderParam kernel_param_heuristic;
};
} // namespace miopen
#endif
