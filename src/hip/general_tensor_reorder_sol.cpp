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

#include <miopen/general_tensor_reorder_sol.hpp>
#include <miopen/tensor.hpp>
#include <miopen/magic_div.hpp>
#include <miopen/float_equal.hpp>
#include <string>
#include <vector>
#include <limits>
#include <iostream>
#include <sstream>
#include <../kernels/gpu_general_tensor_reorder_kernel/order.hpp>

#define TENSOR_REORDER_BLOCK_SIZE 256
#define TENSOR_REORDER_PERSISTENT 0

#if TENSOR_REORDER_PERSISTENT
#define TENSOR_REORDER_OCCUPANCY 4
#endif

namespace miopen {
namespace tensor_reorder {

static inline std::string GetNameTrait(std::size_t type_size)
{
    if(type_size == 1)
        return "byte";
    if(type_size == 2)
        return "half";
    if(type_size == 4)
        return "dword";
    MIOPEN_THROW("data type not supported");
}

template<typename dst_order>
static inline std::string GetKernelName(std::size_t data_size, const GeneralReorderParam* kparam)
{
    std::ostringstream kernel_name;
    std::string type_trait = GetNameTrait(data_size);
    kernel_name << "general_4d_reorder_" << kparam->tile_x << "x" << kparam->tile_y << "_";
    if(!(kparam->pack_x == 1 && kparam->pack_y == 1 && kparam->ediv_x == 1 && kparam->ediv_y == 1))
    {
        kernel_name << "pack_" << kparam->pack_x << "x" << kparam->pack_y << "_ediv_"
                    << kparam->ediv_x << "x" << kparam->ediv_y << "_";
    }
    kernel_name << type_trait<<"_r"<<dst_order::at(0)<<dst_order::at(1)<<dst_order::at(2)<<dst_order::at(3);
    std::cout<<kernel_name.str();
    return kernel_name.str();
}

static inline GeneralReorderParam
HeuristicGet(std::size_t data_size, uint32_t dim_0, uint32_t dim_1, uint32_t dim_2, uint32_t dim_3)
{
    /*
     * TODO:
     * Design a algorithm to determine general tensor reorder tile size.
     */
    GeneralReorderParam default_kernel;
    if(data_size <= 4 && dim_0 >= 1 && dim_1 >= 1 && dim_2 >= 1 && dim_3 >= 1)
    {
        if(dim_3 >= 16)
        {
            return GeneralReorderParam{16, 256, 1, 1, 1, 1};
        }
        else if(dim_3 >= 8)
        {
            return GeneralReorderParam{8, 256, 1, 1, 1, 1};
        }
        else if(dim_3 >= 4)
        {
            return GeneralReorderParam{4, 256, 1, 1, 1, 1};
        }
        else if(dim_3 >= 2)
        {
            return GeneralReorderParam{2, 256, 1, 1, 1, 1};
        }
        else
        {
            return GeneralReorderParam{1, 256, 1, 1, 1, 1};
        }
    }
    else{
        return default_kernel;
    }

}

} // namespace tensor_reorder
template<typename dst_order>
GeneralReorderSolution<dst_order>::GeneralReorderSolution(const ExecutionContext& ctx,
                                                          miopenDataType_t data_type_,
                                                          uint32_t dim_0_,
                                                          uint32_t dim_1_,
                                                          uint32_t dim_2_,
                                                          uint32_t dim_3_)
    : data_type(data_type_), dim_0(dim_0_), dim_1(dim_1_), dim_2(dim_2_), dim_3(dim_3_)
{
    if(data_type == miopenInt8x4 || data_type == miopenDouble)
        MIOPEN_THROW("These data type are not supported");
    num_cu                 = ctx.GetStream().GetMaxComputeUnits();
    std::size_t data_size  = miopen::GetTypeSize(data_type);
    kernel_param_heuristic = tensor_reorder::HeuristicGet(data_size, dim_0, dim_1, dim_2, dim_3);
}

template<typename dst_order>
solver::KernelInfo GeneralReorderSolution<dst_order>::GetKernel() const
{
    std::size_t block_size = TENSOR_REORDER_BLOCK_SIZE;
#if TENSOR_REORDER_PERSISTENT
    std::size_t grid_size = num_cu * TENSOR_REORDER_OCCUPANCY;
#else
    uint32_t pixel_total = dim_0 * dim_1 * dim_2 * dim_3;
    uint32_t dim_total = (pixel_total + block_size * kernel_param_heuristic.tile_x - 1) / (block_size * kernel_param_heuristic.tile_x);
    std::size_t grid_size = dim_total;
#endif
    std::cout<<"check point before GetKernelName"<<std::endl;
    std::string kernel_name = GetKernelName();
    std::cout<<"check point after GetKernelName with:"<<kernel_name<<std::endl;
    solver::KernelInfo kernel;
    kernel.kernel_file = "general_tensor_reorder.cpp";
    kernel.kernel_name = kernel_name;
    kernel.g_wk.clear();
    kernel.g_wk.push_back(grid_size * block_size);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);
    kernel.l_wk.clear();
    kernel.l_wk.push_back(block_size);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    MIOPEN_LOG_I2("GeneralReorderSolution use kernel: " + kernel_name);

    return kernel;
}

template<typename dst_order>
std::vector<OpKernelArg> GeneralReorderSolution<dst_order>::GetKernelArg() const
{
    std::size_t block_size = TENSOR_REORDER_BLOCK_SIZE;
    uint32_t pixel_total = dim_0 * dim_1 * dim_2 * dim_3;
    uint32_t dim_total = (pixel_total + block_size * kernel_param_heuristic.tile_x - 1) / (block_size * kernel_param_heuristic.tile_x);
#if TENSOR_REORDER_PERSISTENT
    std::size_t grid_size = num_cu * TENSOR_REORDER_OCCUPANCY;
#else
    std::size_t grid_size = dim_total;
#endif

    magic_div_u32_t magic_stride0 = magic_div_u32_gen(dim_1 * dim_2 * dim_3);
    magic_div_u32_t magic_stride1 = magic_div_u32_gen(dim_2 * dim_3);
    magic_div_u32_t magic_stride2 = magic_div_u32_gen(dim_3);

    std::vector<OpKernelArg> opArgs;
    opArgs.emplace_back(0); // placeholder
    opArgs.emplace_back(0); // placeholder
    opArgs.emplace_back(dim_0);
    opArgs.emplace_back(dim_1);
    opArgs.emplace_back(dim_2);
    opArgs.emplace_back(dim_3);
    opArgs.emplace_back(static_cast<uint32_t>(grid_size));
    opArgs.emplace_back(dim_total);
    opArgs.emplace_back(magic_stride0.magic);
    opArgs.emplace_back(static_cast<uint32_t>(magic_stride0.shift));
    opArgs.emplace_back(magic_stride1.magic);
    opArgs.emplace_back(static_cast<uint32_t>(magic_stride1.shift));
    opArgs.emplace_back(magic_stride2.magic);
    opArgs.emplace_back(static_cast<uint32_t>(magic_stride2.shift));

    return opArgs;
}

template<typename dst_order>
std::string GeneralReorderSolution<dst_order>::GetKernelName() const
{
    std::size_t data_size = miopen::GetTypeSize(data_type);
    return tensor_reorder::GetKernelName<dst_order>(data_size, &kernel_param_heuristic);
}

template<typename dst_order>
bool GeneralReorderSolution<dst_order>::IsSkippable() const
{
    // Disable the IsSkippable funciton
    return dim_0 == 0 || dim_1 == 0 || dim_2 == 0 || dim_3 == 0 ;
}

template<typename dst_order>
size_t GeneralReorderSolution<dst_order>::GetSize() const
{
    return miopen::GetTypeSize(data_type) * dim_0 * dim_1 * dim_2 * dim_3;
}

//Explicit instance
template struct GeneralReorderSolution<order<0, 1, 3, 2>>;
template struct GeneralReorderSolution<order<0, 2, 1, 3>>;
template struct GeneralReorderSolution<order<0, 2, 3, 1>>;
template struct GeneralReorderSolution<order<0, 3, 1, 2>>;
template struct GeneralReorderSolution<order<0, 3, 2, 1>>;

template struct GeneralReorderSolution<order<1, 0, 2, 3>>;
template struct GeneralReorderSolution<order<1, 0, 3, 2>>;
template struct GeneralReorderSolution<order<1, 2, 0, 3>>;
template struct GeneralReorderSolution<order<1, 2, 3, 0>>;
template struct GeneralReorderSolution<order<1, 3, 0, 2>>;
template struct GeneralReorderSolution<order<1, 3, 2, 0>>;

template struct GeneralReorderSolution<order<2, 0, 1, 3>>;
template struct GeneralReorderSolution<order<2, 0, 3, 1>>;
template struct GeneralReorderSolution<order<2, 1, 0, 3>>;
template struct GeneralReorderSolution<order<2, 1, 3, 0>>;
template struct GeneralReorderSolution<order<2, 3, 0, 1>>;
template struct GeneralReorderSolution<order<2, 3, 1, 0>>;

template struct GeneralReorderSolution<order<3, 0, 1, 2>>;
template struct GeneralReorderSolution<order<3, 0, 2, 1>>;
template struct GeneralReorderSolution<order<3, 1, 0, 2>>;
template struct GeneralReorderSolution<order<3, 1, 2, 0>>;
template struct GeneralReorderSolution<order<3, 2, 0, 1>>;
template struct GeneralReorderSolution<order<3, 2, 1, 0>>;
} // namespace miopen
