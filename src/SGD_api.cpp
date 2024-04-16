#include "miopen/common.hpp"
#include "miopen/tensor.hpp"
#include <algorithm>
#include <miopen/SGD.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>
#include <sstream>
#include <string>

static void LogCmdSGD(const miopenTensorDescriptor_t& praramInDesc, bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(praramInDesc).GetType();
        if (dtype == miopenHalf)
        {
            ss << "SGDfp16";
        }
        else if(dtype == miopenFloat)
        {
            ss << "SGDfp32";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "SGDf16";
        }
        std::string batch_sz;
        auto dims = miopen::deref(praramInDesc).GetLengths();
        for(auto dim : dims)
        {
            batch_sz += std::to_string(dim);
            batch_sz += ",";
        }
        ss << " -dims " << batch_sz;
        ss << " -F " << ((is_fwd) ? "1" : "2");

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t miopenSGDForward(miopenHandle_t handle,
                                              const miopenTensorDescriptor_t paramInDesc,
                                              const void* paramIn,
                                              const miopenTensorDescriptor_t paramOutDesc,
                                              void* paramOut,
                                              const miopenTensorDescriptor_t gradDesc,
                                              const void* grad,
                                              const miopenTensorDescriptor_t momentumBufferInDesc,
                                              const void* momentumBufferIn,
                                              const miopenTensorDescriptor_t momentumBufferOutDesc,
                                              void* momentumBufferOut,
                                              const double lr,
                                              const double momentum,
                                              const double dampening,
                                              const double weightDecay,
                                              const char nesterov,
                                              const char momentumInitialized)
{
    MIOPEN_LOG_FUNCTION(handle, paramInDesc, paramOutDesc, gradDesc, momentumBufferInDesc, momentumBufferOutDesc, lr, momentum, dampening, weightDecay, (int)nesterov, (int)momentumInitialized);
    LogCmdSGD(paramInDesc, true);
    return miopen::try_([&] {

        std::vector<ConstData_t> xCast;
        std::vector<miopen::TensorDescriptor*> xDescCast;
        miopen::SGDForward(miopen::deref(handle),
                           miopen::deref(paramInDesc),
                           DataCast(paramIn),
                           miopen::deref(paramOutDesc),
                           DataCast(paramOut),
                           miopen::deref(gradDesc),
                           DataCast(grad),
                           miopen::deref(momentumBufferInDesc),
                           DataCast(momentumBufferIn),
                           miopen::deref(momentumBufferOutDesc),
                           DataCast(momentumBufferOut),
                           lr,
                           momentum,
                           dampening,
                           weightDecay,
                           nesterov,
                           momentumInitialized);
    }); 

}
