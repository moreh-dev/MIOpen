#include "registry_driver_maker.hpp"
#include "SGD_driver.hpp"

static Driver* makeDriver(const std::string& base_arg)
{
    if(base_arg == "SGD")
        return new SGDDriver<float, float>();
    if(base_arg == "SGDfp16")
        return new SGDDriver<float16, float>();
    if(base_arg == "SGDbfp16")
        return new SGDDriver<bfloat16, float>();
    return nullptr;
}

REGISTER_DRIVER_MAKER(makeDriver);
