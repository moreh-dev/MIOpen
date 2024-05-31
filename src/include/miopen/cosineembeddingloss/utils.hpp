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
#pragma once

#include <miopen/kernel_build_params.hpp>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>
#include <miopen/solver.hpp>

namespace miopen {
namespace solver {
namespace cosineembeddingloss {

size_t get_reqd_work_item_cnt(const ExecutionContext& context, size_t local_size);

size_t get_reqd_work_item_cnt(const Handle& handle, size_t local_size);

size_t get_parallelism_size(size_t reqd_work_item_cnt, size_t output_numel, size_t reduce_size);

bool is_parallelism(size_t reqd_work_item_cnt, size_t output_numel, size_t reduce_size);

} // namespace cosineembeddingloss
} // namespace solver
} // namespace miopen
