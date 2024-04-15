#ifndef MIOPEN_SGD_HPP_
#define MIOPEN_SGD_HPP_

#include <miopen/common.hpp>

namespace miopen {

struct Handle;
struct TensorDescriptor;

miopenStatus_t SGDForward(Handle& handle,
                          const TensorDescriptor& paramInDesc,
                          ConstData_t paramIn,
                          const TensorDescriptor& paramOutDesc,
                          Data_t  paramOut,
                          const TensorDescriptor& gradDesc,
                          ConstData_t grad,
                          const TensorDescriptor& momentumBufferInDesc,
                          ConstData_t momentumBufferIn,
                          const TensorDescriptor& momentumBufferOutDesc,
                          Data_t  momentumBufferOut,
                          double lr,    
                          double momentum,
                          double dampening,    
                          double weightDecay,
                          char nesterov,
                          char momentumInitialized);

} // namespace miopen
#endif // _MIOPEN_SGD_HPP_
