#pragma once

#include <miopen/invoke_params.hpp>
#include <miopen/tensor.hpp>

namespace miopen {
namespace SGD {

struct InvokeParams : public miopen::InvokeParams
{
    InvokeParams() = default;

    const TensorDescriptor* paramInDesc = nullptr;
    ConstData_t paramIn = nullptr;
    const TensorDescriptor* paramOutDesc = nullptr;
    Data_t paramOut = nullptr;
    const TensorDescriptor* gradDesc = nullptr;
    ConstData_t grad = nullptr;
    const TensorDescriptor* momentumBufferInDesc = nullptr;
    ConstData_t momentumBufferIn = nullptr;
    const TensorDescriptor* momentumBufferOutDesc = nullptr;
    Data_t momentumBufferOut = nullptr;
    double lr = 0;
    double momentum = 0;
    double dampening = 0;
    double weightDecay = 0;
    char nesterov = 0;
    char momentum_initialized = 0;

    std::size_t GetWorkspaceSize() const { return 0; }
    Data_t GetWorkspace() const { return nullptr; }
};

} // namespace SGD
} // namespace miopen
