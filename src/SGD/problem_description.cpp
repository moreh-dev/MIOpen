#include <cstdint>
#include <miopen/SGD/problem_description.hpp>
#include <miopen/datatype.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {
namespace SGD {

NetworkConfig ProblemDescription::MakeNetworkConfig() const
{
    auto dtype = paramInDesc.GetType();
    int32_t total_dims = paramInDesc.GetLengths().size();

    int32_t param_size = 0;
    for(int32_t i = 0; i < total_dims; ++i)
    {
        param_size += paramInDesc.GetLengths()[i];
    }

    std::ostringstream ss;
    ss << "dtype" << dtype;
    ss << "total_dims" << total_dims;
    ss << "param_size" << param_size;

    return NetworkConfig{ss.str()};
}

} // namespace SGD
} // namespace miopen
