#include "miopen/sgd/problem_description.hpp"
#include "miopen/invoke_params.hpp"
#include "miopen/miopen.h"
#include "miopen/names.hpp"
#include "miopen/reduce/invoke_params.hpp"
#include "miopen/reducetensor.hpp"
#include <miopen/sgd.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/check_numerics.hpp>
#include <miopen/tensor.hpp>
#include <miopen/datatype.hpp>
#include <miopen/sgd/invoke_params.hpp>
#include <miopen/sgd/solvers.hpp>
#include <miopen/find_solution.hpp>

namespace miopen {

miopenStatus_t SGDForward(Handle& handle,
                          const TensorDescriptor& paramInDesc,
                          ConstData_t paramIn,
                          const TensorDescriptor& paramOutDesc,
                          Data_t paramOut,
                          const TensorDescriptor& gradDesc,
                          ConstData_t grad,
                          const TensorDescriptor& momentumBufferInDesc,
                          ConstData_t momentumBufferIn,
                          const TensorDescriptor& momentumBufferOutDesc,
                          Data_t momentumBufferOut,
                          double lr,
                          double momentum,
                          double dampening,
                          double weightDecay,
                          char nesterov,
                          char momentum_initialized)
{
    const auto problem = SGD::ProblemDescription{paramInDesc,
                                                 paramOutDesc,
                                                 gradDesc,
                                                 momentumBufferInDesc,
                                                 momentumBufferOutDesc,
                                                 lr,
                                                 momentum,
                                                 dampening,
                                                 weightDecay,
                                                 nesterov,
                                                 momentum_initialized};

    std::vector<size_t> dims_h                  = paramInDesc.GetLengths();
    miopen::Allocator::ManageDataPtr dims_d_ptr = handle.Write(dims_h);
    ConstData_t dims_d                          = dims_d_ptr.get();

    std::vector<size_t> strides_h                  = paramInDesc.GetStrides();
    miopen::Allocator::ManageDataPtr strides_d_ptr = handle.Write(strides_h);
    ConstData_t stride_d                           = strides_d_ptr.get();

    const auto invoke_params = [&]() {
        auto tmp                  = SGD::InvokeParams{};
        tmp.type                  = InvokeType::Run;
        tmp.paramInDesc           = &paramInDesc;
        tmp.paramIn               = paramIn;
        tmp.paramOutDesc          = &paramOutDesc;
        tmp.paramOut              = paramOut;
        tmp.gradDesc              = &gradDesc;
        tmp.grad                  = grad;
        tmp.momentumBufferInDesc  = &momentumBufferInDesc;
        tmp.momentumBufferIn      = momentumBufferIn;
        tmp.momentumBufferOutDesc = &momentumBufferOutDesc;
        tmp.momentumBufferOut     = momentumBufferOut;
        tmp.lr                    = lr;
        tmp.momentum              = momentum;
        tmp.dampening             = dampening;
        tmp.weightDecay           = weightDecay;
        tmp.nesterov              = nesterov;
        tmp.momentum_initialized  = momentum_initialized;
        tmp.dims                  = dims_d;
        tmp.strides               = stride_d;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"SGDForward"};
    const auto solvers = solver::SolverContainer<solver::SGD::SGDForward>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
