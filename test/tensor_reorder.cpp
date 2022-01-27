/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
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
#include <miopen/handle.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor_reorder_util.hpp>
#include <miopen/tensor.hpp>
#include <miopen/tensor_layout.hpp>
//#include <miopen/general_tensor_reorder_sol.hpp>
#include <miopen/invoker.hpp>
#include <miopen/invoke_params.hpp>
#include <boost/optional.hpp>
#include <vector>
#include <cstdlib>
#include <ctime>
#include "test.hpp"
#include "driver.hpp"
#include "random.hpp"
#include "order.hpp"


template <>
struct miopen_type<uint8_t> : std::integral_constant<miopenDataType_t, miopenInt8>
{
};

template <>
struct miopen_type<uint16_t> : std::integral_constant<miopenDataType_t, miopenHalf>
{
};

template<typename T,
         typename dst_order>
void cpu_tensor_reorder(T * dst, T * src, uint64_t dim_0, uint64_t dim_1, uint64_t dim_2, uint64_t dim_3)
{
    constexpr auto dorder = dst_order{};
    const uint64_t src_dim[4] = {dim_0, dim_1, dim_2, dim_3};
    const uint64_t dst_dim[4] = {src_dim[dorder.at(0)], src_dim[dorder.at(1)], src_dim[dorder.at(2)], src_dim[dorder.at(3)]};
    
    const uint64_t src_stride[4]   ={src_dim[1] * src_dim[2] * src_dim[3], 
                                     src_dim[2] * src_dim[3], 
                                     src_dim[3],
                                     1 };
    const uint64_t dst_stride[4]  = {dst_dim[1] * dst_dim[2] * dst_dim[3], 
                                     dst_dim[2] * dst_dim[3], 
                                     dst_dim[3],
                                     1 };

    uint64_t itr_src_dim[4] = {0, 0, 0, 0};
    uint64_t itr_dst_dim[4] = {0, 0, 0, 0};

    for(itr_src_dim[0] = 0; itr_src_dim[0] < src_dim[0]; itr_src_dim[0]++){
        for(itr_src_dim[1] = 0; itr_src_dim[1] < src_dim[1]; itr_src_dim[1]++){
            for(itr_src_dim[2] = 0; itr_src_dim[2] < src_dim[2]; itr_src_dim[2]++){
                for(itr_src_dim[3] = 0; itr_src_dim[3] < src_dim[3]; itr_src_dim[3]++){
                    itr_dst_dim[0] = itr_src_dim[dorder.at(0)];
                    itr_dst_dim[1] = itr_src_dim[dorder.at(1)];
                    itr_dst_dim[2] = itr_src_dim[dorder.at(2)];
                    itr_dst_dim[3] = itr_src_dim[dorder.at(3)];

                    uint64_t idx_src =   itr_src_dim[0] * src_stride[0] +
                                         itr_src_dim[1] * src_stride[1] +
                                         itr_src_dim[2] * src_stride[2] +
                                         itr_src_dim[3] * src_stride[3] ;
                    uint64_t idx_dst =   itr_dst_dim[0] * dst_stride[0] + 
                                         itr_dst_dim[1] * dst_stride[1] +
                                         itr_dst_dim[2] * dst_stride[2] +
                                         itr_dst_dim[3] * dst_stride[3] ;
                    
                    dst[idx_dst] = src[idx_src]; 
                }
            }
        }
    }
}

template <typename T, typename dst_order>
struct cpu_reorder
{
    static void run(T* dst, T* src, uint64_t N, uint64_t C, uint64_t H, uint64_t W)
    {
        cpu_tensor_reorder<T, dst_order>(dst, src, N, C, H, W);
    }
};

template <typename dst_order>
struct reorder_str
{
    static std::string get() { 
        return ("r" + std::to_string(dst_order::at(0)) 
                    + std::to_string(dst_order::at(1))
                    + std::to_string(dst_order::at(2))
                    + std::to_string(dst_order::at(3)) ); 
        }
};

enum tensor_layout_t
{
    miopen_tensor_layout_nchw,
    miopen_tensor_layout_ncdhw,
    miopen_tensor_layout_nhwc,
    miopen_tensor_layout_ndhwc,
};

std::string tensor_layout_to_string(tensor_layout_t layout)
{
    std::string layout_string("N/A");
    if(layout == miopen_tensor_layout_nchw)
        layout_string = "NCHW";
    else if(layout == miopen_tensor_layout_ncdhw)
        layout_string = "NCDHW";
    else if(layout == miopen_tensor_layout_nhwc)
        layout_string = "NHWC";
    else if(layout == miopen_tensor_layout_ndhwc)
        layout_string = "NDHWC";
    else
        MIOPEN_THROW("Unsupported tensor layout");
    return layout_string;
}


template <typename T>
struct to_miopen_data_type
{
};

template <>
struct to_miopen_data_type<float>
{
    static miopenDataType_t get() { return miopenFloat; }
};

template <>
struct to_miopen_data_type<uint16_t>
{
    static miopenDataType_t get() { return miopenHalf; } // we actually didn't calculate 16bit float
};

template <>
struct to_miopen_data_type<uint8_t>
{
    static miopenDataType_t get() { return miopenInt8; }
};

#define RAND_INTEGER_MAX 120
#define RAND_INTEGER_MIN -88

static int gen_rand_integer()
{
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static int inited = 0;
    if(inited == 0)
    {
        std::srand(std::time(nullptr));
        inited = 1;
    }
    return GET_RAND();
}

template <typename T>
void rand_tensor_integer(tensor<T>& t, int max = RAND_INTEGER_MAX, int min = RAND_INTEGER_MIN)
{
    // use integer to random.
    for(int i = 0; i < t.data.size(); i++)
        t[i] = static_cast<T>(gen_rand_integer() % (max - min) + min);
}

template <typename T>
bool compare_equal(T r1, T r2)
{
    return r1 == r2;
}

template <>
bool compare_equal<float>(float r1, float r2)
{
    return miopen::float_equal(r1, r2);
}

template <typename T>
bool verify_tensor(tensor<T>& t_gpu, tensor<T>& t_cpu)
{
    if(t_gpu.data.size() != t_cpu.data.size())
    {
        MIOPEN_LOG_E("size not equal, should not happen");
        return false;
    }
    auto idx          = miopen::mismatch_idx(t_gpu.data, t_cpu.data, compare_equal<T>);
    bool valid_result = idx >= miopen::range_distance(t_cpu);

    if(!valid_result)
    {
        std::cout << "diff at:" << idx << ", gpu:" << t_gpu[idx] << ", cpu:" << t_cpu[idx]
                  << std::endl;
    }
    return valid_result;
}

//compile time for_loop
namespace detail {

    template<class T, T... inds, class F>
    constexpr void loop(std::integer_sequence<T, inds...>, F&& f) {
        (f(std::integral_constant<T, inds>{}), ...);// C++17 fold expression
    }

}

template<class T, T count, class F>
constexpr void loop(F&& f) {
    detail::loop(std::make_integer_sequence<T, count>{}, std::forward<F>(f));
}

struct reorder_base
{
    miopenHandle_t handle{};
#if MIOPEN_BACKEND_OPENCL
    cl_command_queue q{};
#endif

    reorder_base()
    {
        miopenCreate(&handle);
#if MIOPEN_BACKEND_OPENCL
        miopenGetStream(handle, &q);
#endif
    }
    ~reorder_base() { miopenDestroy(handle); }

    static std::vector<uint32_t> get_dim_3_size() { return {1, 9, 14}; }
    static std::vector<uint32_t> get_dim_2_size() { return {1, 9, 14}; }
    static std::vector<uint32_t> get_dim_1_size() { return {3, 8, 14}; }
    static std::vector<uint32_t> get_dim_0_size() { return {1, 2}; }

    template <typename F>
    void iterate_reorder(F f)
    {
        std::vector<uint32_t> dim_3_list = get_dim_3_size();
        std::vector<uint32_t> dim_2_list = get_dim_2_size();
        std::vector<uint32_t> dim_1_list = get_dim_1_size();
        std::vector<uint32_t> dim_0_list = get_dim_0_size();
        
        dim_3_list.push_back(gen_rand_integer() % 13 + 29);
        dim_2_list.push_back(gen_rand_integer() % 13 + 29);
        dim_1_list.push_back(gen_rand_integer() % 13 + 15);
        dim_0_list.push_back(gen_rand_integer() % 4 + 3);

        for(uint32_t dim_3 : dim_3_list)
        {
            for(uint32_t dim_2 : dim_2_list)
            {
                for(uint32_t dim_1 : dim_1_list)
                {
                    for(uint32_t dim_0 : dim_0_list)
                    {
                        f(dim_0, dim_1, dim_2, dim_3);
                    }
                }
            }
        }
    }
};

struct reorder_invoke_param : public miopen::InvokeParams
{
    ConstData_t src = nullptr;
    Data_t dst      = nullptr;

    reorder_invoke_param(ConstData_t src_, Data_t dst_) : src(src_), dst(dst_) {}
    reorder_invoke_param(miopen::InvokeType type_, ConstData_t src_, Data_t dst_)
        : InvokeParams{type_}, src(src_), dst(dst_)
    {
    }
};
//The template parameter dst_order is just for CPU verification
template <typename T, typename dst_order, typename REORDER_SOL>
struct reorder_test : reorder_base
{
    void run()
    {
        auto run_reorder = [this](uint32_t dim_0, uint32_t dim_1, uint32_t dim_2, uint32_t dim_3) {
            int tensor_sz = dim_0 * dim_1 * dim_2 * dim_3;
            std::vector<int> tensor_len({static_cast<int>(dim_0),
                                         static_cast<int>(dim_1),
                                         static_cast<int>(dim_2),
                                         static_cast<int>(dim_3)});

            std::vector<int> tensor_strides;

            std::string layout_default = miopen::tensor_layout_get_default(4);
            std::string layout_string  = tensor_layout_to_string(miopen_tensor_layout_nchw);

            miopen::tensor_layout_to_strides(
                tensor_len, layout_default, layout_string, tensor_strides);

            tensor<T> t_src(tensor_len, tensor_strides);
            tensor<T> t_dst(tensor_len, tensor_strides);
            tensor<T> t_dst_gpu(tensor_len, tensor_strides);
            rand_tensor_integer(t_src);
#if MIOPEN_BACKEND_OPENCL
            cl_context cl_ctx;
            clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &cl_ctx, nullptr);
            cl_int status = CL_SUCCESS;
            cl_mem src_dev =
                clCreateBuffer(cl_ctx, CL_MEM_READ_WRITE, sizeof(T) * tensor_sz, nullptr, &status);
            cl_mem dst_dev =
                clCreateBuffer(cl_ctx, CL_MEM_READ_WRITE, sizeof(T) * tensor_sz, nullptr, nullptr);
            status |= clEnqueueWriteBuffer(q,
                                           src_dev,
                                           CL_TRUE,
                                           0,
                                           sizeof(T) * tensor_sz,
                                           t_src.data.data(),
                                           0,
                                           nullptr,
                                           nullptr);
            EXPECT(status == CL_SUCCESS);
#elif MIOPEN_BACKEND_HIP
            void* src_dev;
            void* dst_dev;
            EXPECT(hipMalloc(&src_dev, sizeof(T) * tensor_sz) == hipSuccess);
            EXPECT(hipMalloc(&dst_dev, sizeof(T) * tensor_sz) == hipSuccess);
            EXPECT(hipMemcpy(
                       src_dev, t_src.data.data(), sizeof(T) * tensor_sz, hipMemcpyHostToDevice) ==
                   hipSuccess);
#endif

            const auto invoke_param = reorder_invoke_param{
                DataCast(static_cast<const void*>(src_dev)), DataCast(dst_dev)};

            miopen::ExecutionContext ctx;
            ctx.SetStream(&miopen::deref(this->handle));
            ctx.DetectRocm();
            // ctx.SetupFloats();

            REORDER_SOL reorder_sol(ctx, to_miopen_data_type<T>::get(), dim_0, dim_1, dim_2, dim_3);

            std::vector<OpKernelArg> opArgs = reorder_sol.GetKernelArg();

            boost::optional<miopen::InvokerFactory> invoker_factory(
                [=](const std::vector<miopen::Kernel>& kernels) mutable {
                    return [=](const miopen::Handle& handle,
                               const miopen::AnyInvokeParams& primitive_param) mutable {
                        decltype(auto) invoke_params =
                            primitive_param.CastTo<reorder_invoke_param>();

                        const auto k = handle.Run(kernels[0]);

                        opArgs[0] = OpKernelArg(invoke_params.dst);
                        opArgs[1] = OpKernelArg(invoke_params.src);

                        k(opArgs);
                    };
                });

            std::vector<miopen::solver::KernelInfo> construction_params{reorder_sol.GetKernel()};

            const auto invoker =
                miopen::deref(this->handle).PrepareInvoker(*invoker_factory, construction_params);

            // run gpu
            invoker(miopen::deref(this->handle), invoke_param);

            // run cpu
            cpu_reorder<T, dst_order>::run(t_dst.data.data(), t_src.data.data(), dim_0, dim_1, dim_2, dim_3);

#if MIOPEN_BACKEND_OPENCL
            status = clEnqueueReadBuffer(q,
                                         dst_dev,
                                         CL_TRUE,
                                         0,
                                         sizeof(T) * tensor_sz,
                                         t_dst_gpu.data.data(),
                                         0,
                                         nullptr,
                                         nullptr);
            EXPECT(status == CL_SUCCESS);
#elif MIOPEN_BACKEND_HIP
            EXPECT(hipMemcpy(t_dst_gpu.data.data(),
                             dst_dev,
                             sizeof(T) * tensor_sz,
                             hipMemcpyDeviceToHost) == hipSuccess);
#endif

            // we expect excact match, since use integer
            bool valid_result = verify_tensor(t_dst_gpu, t_dst);

            std::cout << "[" << reorder_str<dst_order>::get() << ", b" << (sizeof(T) * 8)
                      << " ] "
                      << "dim_0:" << dim_0 << ", dim_1:" << dim_1 << ", dim_2:" << dim_2 << ", dim_3:" << dim_3
                      << ", valid:" << valid_result << std::endl;

            EXPECT(valid_result == true);

#if MIOPEN_BACKEND_OPENCL
            clReleaseMemObject(src_dev);
            clReleaseMemObject(dst_dev);
#elif MIOPEN_BACKEND_HIP
            hipFree(src_dev);
            hipFree(dst_dev);
#endif
        };

        iterate_reorder(run_reorder);
    }
};


int main()
{
loop<int, 1>([&](auto i) {
    constexpr int all_possible_sequence[23][4] = {
    {0, 3, 2, 1}, {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 3, 1}, {0, 3, 1, 2}, 
    {1, 0, 2, 3}, {1, 0, 3, 2}, {1, 2, 0, 3}, {1, 2, 3, 0}, {1, 3, 0, 2}, {1, 3, 2, 0},
    {2, 0, 1, 3}, {2, 0, 3, 1}, {2, 1, 0, 3}, {2, 1, 3, 0}, {2, 3, 0, 1}, {2, 3, 1, 0},
    {3, 0, 1, 2}, {3, 0, 2, 1}, {3, 1, 0, 2}, {3, 1, 2, 0}, {3, 2, 0, 1}, {3, 2, 1, 0} };
    using dst_order = order<all_possible_sequence[i][0], all_possible_sequence[i][1], all_possible_sequence[i][2], all_possible_sequence[i][3]>;
    run_test<reorder_test<float,    dst_order, miopen::TensorReorderSolution<dst_order> >>();
    run_test<reorder_test<uint16_t, dst_order, miopen::TensorReorderSolution<dst_order> >>();
    run_test<reorder_test<uint8_t,  dst_order, miopen::TensorReorderSolution<dst_order> >>();
});
}