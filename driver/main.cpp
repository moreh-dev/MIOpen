#include <iostream>
#include <cstdio>

#include "InputFlags.hpp"

#include <miopen/miopen.h>

#include <miopen/any_solver.hpp>
#include <miopen/tensor_layout.hpp>
#include <miopen/find_db.hpp>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/range/adaptors.hpp>

#define USE_CUSTOM_API 0

using namespace miopen;

// globals
InputFlags inflags;
int convDirection;
int key_in_channels           = 0;
int key_out_channels          = 0;
std::vector<int> key_out_lens = {0, 0, 0};
miopenDataType_t data_type;

miopenConvolutionDescriptor_t convDesc;

miopenTensorDescriptor_t inputTensor;
miopenTensorDescriptor_t weightTensor;
miopenTensorDescriptor_t outputTensor;
miopenTensorDescriptor_t biasTensor;

int AddCmdLineArgs()
{
    inflags.AddInputFlag("in_layout",
                         'I',
                         "",
                         "Input Layout (Default=NCHW for 2d conv, NCDHW for 3d conv)",
                         "string",
                         true);
    inflags.AddInputFlag("out_layout",
                         'O',
                         "",
                         "Output Layout (Default=NCHW for 2d conv, NCDHW for 3d conv)",
                         "string",
                         true);
    inflags.AddInputFlag("fil_layout",
                         'f',
                         "",
                         "Filter Layout (Default=NCHW for 2d conv, NCDHW for 3d conv)",
                         "string",
                         true);
    inflags.AddInputFlag(
        "spatial_dim", '_', "2", "convolution spatial dimension (Default-2)", "int");
    inflags.AddInputFlag("forw",
                         'F',
                         "0",
                         "Flag enables fwd, bwd, wrw convolutions"
                         "\n0 fwd+bwd+wrw (default)"
                         "\n1 fwd only"
                         "\n2 bwd only"
                         "\n4 wrw only"
                         "\n3 fwd+bwd"
                         "\n5 fwd+wrw"
                         "\n6 bwd+wrw",
                         "int");
    inflags.AddInputFlag("batchsize", 'n', "100", "Mini-batch size (Default=100)", "int");
    inflags.AddInputFlag("in_channels", 'c', "3", "Number of Input Channels (Default=3)", "int");
    inflags.AddInputFlag("in_d", '!', "32", "Input Depth (Default=32)", "int");
    inflags.AddInputFlag("in_h", 'H', "32", "Input Height (Default=32)", "int");
    inflags.AddInputFlag("in_w", 'W', "32", "Input Width (Default=32)", "int");
    inflags.AddInputFlag(
        "out_channels", 'k', "32", "Number of Output Channels (Default=32)", "int");
    inflags.AddInputFlag("fil_d", '@', "3", "Filter Depth (Default=3)", "int");
    inflags.AddInputFlag("fil_h", 'y', "3", "Filter Height (Default=3)", "int");
    inflags.AddInputFlag("fil_w", 'x', "3", "Filter Width (Default=3)", "int");
    inflags.AddInputFlag(
        "conv_stride_d", '#', "1", "Convolution Stride for Depth (Default=1)", "int");
    inflags.AddInputFlag(
        "conv_stride_h", 'u', "1", "Convolution Stride for Height (Default=1)", "int");
    inflags.AddInputFlag(
        "conv_stride_w", 'v', "1", "Convolution Stride for Width (Default=1)", "int");
    inflags.AddInputFlag("pad_d", '$', "0", "Zero Padding for Depth (Default=0)", "int");
    inflags.AddInputFlag("pad_h", 'p', "0", "Zero Padding for Height (Default=0)", "int");
    inflags.AddInputFlag("pad_w", 'q', "0", "Zero Padding for Width (Default=0)", "int");
    inflags.AddInputFlag("pad_val", 'r', "0", "Padding Value (Default=0)", "int");
    inflags.AddInputFlag(
        "trans_output_pad_d", '%', "0", "Zero Padding Output for Depth (Default=0)", "int");
    inflags.AddInputFlag(
        "trans_output_pad_h", 'Y', "0", "Zero Padding Output for Height (Default=0)", "int");
    inflags.AddInputFlag(
        "trans_output_pad_w", 'X', "0", "Zero Padding Output for Width (Default=0)", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("verification_cache",
                         'C',
                         "",
                         "Use specified directory to cache verification data. Off by default.",
                         "string");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag("wall",
                         'w',
                         "0",
                         "Wall-clock Time Each Layer"
                         "\n0 Off (Default)"
                         "\n1 On, requires '--time 1')"
                         "\n2 On, warm-up the library (prefetch db caches), requires '--time 1')",
                         "int");
    inflags.AddInputFlag("search", 's', "0", "Search Kernel Config (Default=0)", "int");
    inflags.AddInputFlag("printconv", 'P', "1", "Print Convolution Dimensions (Default=1)", "int");
    inflags.AddInputFlag("dump_output", 'o', "0", "Dumps the output buffers (Default=0)", "int");
    inflags.AddInputFlag("in_data", 'd', "", "Input data filename (Default=)", "string");
    inflags.AddInputFlag("weights", 'e', "", "Input weights filename (Default=)", "string");
    inflags.AddInputFlag("bias", 'b', "", "Use Bias (Default=0)", "int");
    inflags.AddInputFlag(
        "mode", 'm', "conv", "Convolution Mode (conv, trans) (Default=conv)", "str");

    inflags.AddInputFlag(
        "pad_mode", 'z', "default", "Padding Mode (same, valid, default) (Default=default)", "str");
    inflags.AddInputFlag("tensor_vect",
                         'Z',
                         "0",
                         "tensor vectorization type (none, vect_c, vect_n) (Default=0)",
                         "int");
    inflags.AddInputFlag(
        "vector_length", 'L', "1", "tensor vectorization length (Default=1)", "int");
    inflags.AddInputFlag("dilation_d", '^', "1", "Dilation of Filter Depth (Default=1)", "int");
    inflags.AddInputFlag("dilation_h", 'l', "1", "Dilation of Filter Height (Default=1)", "int");
    inflags.AddInputFlag("dilation_w", 'j', "1", "Dilation of Filter Width (Default=1)", "int");
    inflags.AddInputFlag("in_bias", 'a', "", "Input bias filename (Default=)", "string");
    inflags.AddInputFlag("group_count", 'g', "1", "Number of Groups (Default=1)", "int");
    inflags.AddInputFlag("dout_data",
                         'D',
                         "",
                         "dy data filename for backward weight computation (Default=)",
                         "string");
    inflags.AddInputFlag("solution",
                         'S',
                         "-1",
                         "Use immediate mode, run solution with specified id."
                         "\nAccepts integer argument N:"
                         "\n=0 Immediate mode, build and run fastest solution"
                         "\n>0 Immediate mode, build and run solution_id = N"
                         "\n<0 Use Find() API (Default=-1)"
                         "\nAlso accepts symbolic name of solution:"
                         "\n<valid name>   Immediate mode, build and run specified solution"
                         "\n<invalid name> Use Find() API",
                         "string");

    return 0;
}

void ParseKey(std::string key)
{
    char h_sep = '-';
    char x_sep = 'x';

    int h_pos;
    int x_pos;

    int n;

    std::string contents, buf;

    // GetInChannels
    h_pos           = key.find(h_sep);
    key_in_channels = std::stoi(key.substr(0, h_pos));
    inflags.AddInputFlag(
        "in_channels", 'c', key.substr(0, h_pos), "Number of Input Channels (Default=3)", "int");

    contents = key.substr(h_pos + 1);

    // GetInDHW
    x_pos                = contents.find(x_sep);
    buf                  = contents.substr(0, x_pos);
    n                    = std::count(buf.begin(), buf.end(), h_sep);
    bool spatial_dim_set = false;

    if(n == 3)
    {
        inflags.AddInputFlag(
            "spatial_dim", '_', "3", "convolution spatial dimension (Default-2)", "int");
        spatial_dim_set = true;

        h_pos = contents.find(h_sep);
        inflags.AddInputFlag(
            "in_d", '!', contents.substr(0, h_pos), "Input Depth (Default=32)", "int");
        contents = contents.substr(h_pos + 1);
    }
    if(!spatial_dim_set)
        inflags.AddInputFlag(
            "spatial_dim", '_', "2", "convolution spatial dimension (Default-2)", "int");

    h_pos = contents.find(h_sep);
    inflags.AddInputFlag(
        "in_h", 'H', contents.substr(0, h_pos), "Input Height (Default=32)", "int");
    contents = contents.substr(h_pos + 1);

    h_pos = contents.find(h_sep);
    inflags.AddInputFlag("in_w", 'W', contents.substr(0, h_pos), "Input Width (Default=32)", "int");
    contents = contents.substr(h_pos + 1);

    // GetWeightDHW
    h_pos = contents.find(h_sep);
    buf   = contents.substr(0, h_pos);
    n     = std::count(buf.begin(), buf.end(), x_sep);

    if(n == 2)
    {
        x_pos = contents.find(x_sep);
        inflags.AddInputFlag(
            "fil_d", '@', contents.substr(0, x_pos), "Filter Depth (Default=3)", "int");
        contents = contents.substr(x_pos + 1);
    }
    x_pos = contents.find(x_sep);
    inflags.AddInputFlag(
        "fil_h", 'y', contents.substr(0, x_pos), "Filter Height (Default=3)", "int");
    contents = contents.substr(x_pos + 1);

    h_pos = contents.find(h_sep);
    inflags.AddInputFlag(
        "fil_w", 'x', contents.substr(0, h_pos), "Filter Width (Default=3)", "int");
    contents = contents.substr(h_pos + 1);

    // GetOutChannels
    h_pos            = contents.find(h_sep);
    key_out_channels = std::stoi(contents.substr(0, h_pos));
    inflags.AddInputFlag("out_channels",
                         'k',
                         contents.substr(0, h_pos),
                         "Number of Output Channels (Default=32)",
                         "int");
    contents = contents.substr(h_pos + 1);

    // GetOutDHW
    x_pos = contents.find(x_sep);
    buf   = contents.substr(0, x_pos);
    n     = std::count(buf.begin(), buf.end(), h_sep);

    if(n == 4)
    {
        // n include's batchsize's hyphen
        h_pos           = contents.find(h_sep);
        key_out_lens[0] = std::stoi(contents.substr(0, h_pos));
        contents        = contents.substr(h_pos + 1);
    }
    h_pos           = contents.find(h_sep);
    key_out_lens[1] = std::stoi(contents.substr(0, h_pos));
    contents        = contents.substr(h_pos + 1);

    h_pos           = contents.find(h_sep);
    key_out_lens[2] = std::stoi(contents.substr(0, h_pos));
    contents        = contents.substr(h_pos + 1);

    // GetInBatchSize
    h_pos = contents.find(h_sep);
    inflags.AddInputFlag(
        "batchsize", 'n', contents.substr(0, h_pos), "Mini-batch size (Default=100)", "int");
    contents = contents.substr(h_pos + 1);

    // GetPadDHW
    h_pos = contents.find(h_sep);
    buf   = contents.substr(0, h_pos);
    n     = std::count(buf.begin(), buf.end(), x_sep);

    if(n == 2)
    {
        // spatial_dims > 2
        x_pos = contents.find(x_sep);
        inflags.AddInputFlag(
            "pad_d", '$', contents.substr(0, x_pos), "Zero Padding for Depth (Default=0)", "int");
        contents = contents.substr(x_pos + 1);
    }
    x_pos = contents.find(x_sep);
    inflags.AddInputFlag(
        "pad_h", 'p', contents.substr(0, x_pos), "Zero Padding for Height (Default=0)", "int");
    contents = contents.substr(x_pos + 1);

    h_pos = contents.find(h_sep);
    inflags.AddInputFlag(
        "pad_w", 'q', contents.substr(0, h_pos), "Zero Padding for Width (Default=0)", "int");
    contents = contents.substr(h_pos + 1);

    // GetStrideDHW
    h_pos = contents.find(h_sep);
    buf   = contents.substr(0, h_pos);
    n     = std::count(buf.begin(), buf.end(), x_sep);

    if(n == 2)
    {
        // spatial_dims > 2
        x_pos = contents.find(x_sep);
        inflags.AddInputFlag("conv_stride_d",
                             '#',
                             contents.substr(0, x_pos),
                             "Convolution Stride for Depth (Default=1)",
                             "int");
        contents = contents.substr(x_pos + 1);
    }
    x_pos = contents.find(x_sep);
    inflags.AddInputFlag("conv_stride_h",
                         'u',
                         contents.substr(0, x_pos),
                         "Convolution Stride for Height (Default=1)",
                         "int");
    contents = contents.substr(x_pos + 1);

    h_pos = contents.find(h_sep);
    inflags.AddInputFlag("conv_stride_w",
                         'v',
                         contents.substr(0, h_pos),
                         "Convolution Stride for Width (Default=1)",
                         "int");
    contents = contents.substr(h_pos + 1);

    // GetDilationDHW
    h_pos = contents.find(h_sep);
    buf   = contents.substr(0, h_pos);
    n     = std::count(buf.begin(), buf.end(), x_sep);

    if(n == 2)
    {
        // spatial_dims > 2
        x_pos = contents.find(x_sep);
        inflags.AddInputFlag("dilation_d",
                             '^',
                             contents.substr(0, x_pos),
                             "Dilation of Filter Depth (Default=1)",
                             "int");
        contents = contents.substr(x_pos + 1);
    }
    x_pos = contents.find(x_sep);
    inflags.AddInputFlag("dilation_h",
                         'l',
                         contents.substr(0, x_pos),
                         "Dilation of Filter Height (Default=1)",
                         "int");
    contents = contents.substr(x_pos + 1);

    h_pos = contents.find(h_sep);
    inflags.AddInputFlag("dilation_w",
                         'j',
                         contents.substr(0, h_pos),
                         "Dilation of Filter Width (Default=1)",
                         "int");
    contents = contents.substr(h_pos + 1);

    // GetBias
    h_pos = contents.find(h_sep);
    inflags.AddInputFlag("bias", 'b', contents.substr(0, h_pos), "Use Bias (Default=0)", "int");
    contents = contents.substr(h_pos + 1);

    // GetLayouts
    h_pos = contents.find(h_sep);
    inflags.AddInputFlag("in_layout",
                         'I',
                         contents.substr(0, h_pos),
                         "Input Layout (Default=NCHW for 2d conv, NCDHW for 3d conv)",
                         "string",
                         true);
    std::string in_layout = contents.substr(0, h_pos);
    contents              = contents.substr(h_pos + 1);

    h_pos = contents.find(h_sep);
    // if contents' first character is 'N', more layout coming
    if(contents[0] == 'N')
    {
        inflags.AddInputFlag("out_layout",
                             'O',
                             contents.substr(0, h_pos),
                             "Output Layout (Default=NCHW for 2d conv, NCDHW for 3d conv)",
                             "string",
                             true);
        contents = contents.substr(h_pos + 1);

        h_pos = contents.find(h_sep);
        inflags.AddInputFlag("fil_layout",
                             'f',
                             contents.substr(0, h_pos),
                             "Filter Layout (Default=NCHW for 2d conv, NCDHW for 3d conv)",
                             "string",
                             true);
        contents = contents.substr(h_pos + 1);
    }
    else
    {
        inflags.AddInputFlag("out_layout",
                             'O',
                             in_layout,
                             "Output Layout (Default=NCHW for 2d conv, NCDHW for 3d conv)",
                             "string",
                             true);
        inflags.AddInputFlag("fil_layout",
                             'f',
                             in_layout,
                             "Filter Layout (Default=NCHW for 2d conv, NCDHW for 3d conv)",
                             "string",
                             true);
    }

    // GetDataType
    h_pos = contents.find(h_sep);
    if(contents.substr(0, h_pos).compare("FP32") == 0)
    {
        data_type = miopenFloat;
    }
    else if(contents.substr(0, h_pos).compare("FP16") == 0)
    {
        data_type = miopenHalf;
    }
    else if(contents.substr(0, h_pos).compare("BF16") == 0)
    {
        data_type = miopenBFloat16;
    }
    else
    {
        std::cerr << "Invalid data type" << std::endl;
        exit(1);
    }

    contents = contents.substr(h_pos + 1);

    // GetDirection
    if(contents[0] == 'F')
    {
        convDirection = 0;
    }
    else if(contents[0] == 'B')
    {
        convDirection = 1;
    }
    else if(contents[0] == 'W')
    {
        convDirection = 2;
    }
    else
    {
        std::cerr << "Invalid direction" << std::endl;
        exit(1);
    }

    contents = contents.substr(1);

    // GetGroupCount
    // optional
    if(contents.length() != 0)
    {
        inflags.AddInputFlag(
            "group_count", 'g', contents.substr(2), "Number of Groups (Default=1)", "int");
    }
}

std::vector<int> GetInputTensorLengthsFromCmdLine()
{
    std::vector<int> in_lens;

    int spatial_dim = inflags.GetValueInt("spatial_dim");
    in_lens.resize(2 + spatial_dim);

    in_lens[0] = inflags.GetValueInt("batchsize");
    in_lens[1] = inflags.GetValueInt("in_channels");

    auto in_spatial_lens = boost::adaptors::slice(in_lens, 2, 2 + spatial_dim);

    if(spatial_dim == 2)
    {
        in_spatial_lens[0] = inflags.GetValueInt("in_h");
        in_spatial_lens[1] = inflags.GetValueInt("in_w");
    }
    else if(spatial_dim == 3)
    {
        in_spatial_lens[0] = inflags.GetValueInt("in_d");
        in_spatial_lens[1] = inflags.GetValueInt("in_h");
        in_spatial_lens[2] = inflags.GetValueInt("in_w");
    }
    else
    {
        MIOPEN_THROW("unsupported convolution dimension");
    }

    return in_lens;
}

std::vector<int> GetWeightTensorLengthsFromCmdLine()
{
    std::vector<int> wei_lens;

    int spatial_dim = inflags.GetValueInt("spatial_dim");
    wei_lens.resize(2 + spatial_dim);

    auto wei_spatial_lens = boost::adaptors::slice(wei_lens, 2, 2 + spatial_dim);

    int group_count = std::max(inflags.GetValueInt("group_count"), 1);

    int wei_k_len = inflags.GetValueInt("out_channels");
    int wei_c_len = inflags.GetValueInt("in_channels");

    if(spatial_dim == 2)
    {
        wei_spatial_lens[0] = inflags.GetValueInt("fil_h");
        wei_spatial_lens[1] = inflags.GetValueInt("fil_w");
    }
    else if(spatial_dim == 3)
    {
        wei_spatial_lens[0] = inflags.GetValueInt("fil_d");
        wei_spatial_lens[1] = inflags.GetValueInt("fil_h");
        wei_spatial_lens[2] = inflags.GetValueInt("fil_w");
    }
    else
    {
        MIOPEN_THROW("unsupported convolution dimension");
    }

    if(group_count > 1)
    {
        if(wei_c_len % group_count != 0 || wei_k_len % group_count != 0 ||
           group_count > wei_c_len || group_count > wei_k_len)
        {
            MIOPEN_THROW("Invalid group number\n");
        }
    }

    miopenConvolutionMode_t mode;
    if((inflags.GetValueStr("mode")) == "conv")
    {
        mode = miopenConvolution;
    }
    else if((inflags.GetValueStr("mode")) == "trans")
    {
        mode = miopenTranspose;
    }
    else
    {
        MIOPEN_THROW("Incorrect Convolution Mode\n");
    }

    if(mode == miopenTranspose)
    {
        wei_lens[0] = wei_c_len;
        wei_lens[1] = wei_k_len / group_count;
    }
    else
    {
        wei_lens[0] = wei_k_len;
        wei_lens[1] = wei_c_len / group_count;
    }

    return wei_lens;
}

int SetConvDescriptorFromCmdLineArgs()
{
    int spatial_dim = inflags.GetValueInt("spatial_dim");

    std::vector<int> in_spatial_lens(spatial_dim);
    std::vector<int> wei_spatial_lens(spatial_dim);
    std::vector<int> pads(spatial_dim);
    std::vector<int> conv_strides(spatial_dim);
    std::vector<int> conv_dilations(spatial_dim);
    std::vector<int> trans_output_pads(spatial_dim);

    if(spatial_dim == 2)
    {
        in_spatial_lens[0]   = inflags.GetValueInt("in_h");
        in_spatial_lens[1]   = inflags.GetValueInt("in_w");
        wei_spatial_lens[0]  = inflags.GetValueInt("fil_h");
        wei_spatial_lens[1]  = inflags.GetValueInt("fil_w");
        pads[0]              = inflags.GetValueInt("pad_h");
        pads[1]              = inflags.GetValueInt("pad_w");
        conv_strides[0]      = inflags.GetValueInt("conv_stride_h");
        conv_strides[1]      = inflags.GetValueInt("conv_stride_w");
        conv_dilations[0]    = inflags.GetValueInt("dilation_h");
        conv_dilations[1]    = inflags.GetValueInt("dilation_w");
        trans_output_pads[0] = inflags.GetValueInt("trans_output_pad_h");
        trans_output_pads[1] = inflags.GetValueInt("trans_output_pad_w");
    }
    else if(spatial_dim == 3)
    {
        in_spatial_lens[0]   = inflags.GetValueInt("in_d");
        in_spatial_lens[1]   = inflags.GetValueInt("in_h");
        in_spatial_lens[2]   = inflags.GetValueInt("in_w");
        wei_spatial_lens[0]  = inflags.GetValueInt("fil_d");
        wei_spatial_lens[1]  = inflags.GetValueInt("fil_h");
        wei_spatial_lens[2]  = inflags.GetValueInt("fil_w");
        pads[0]              = inflags.GetValueInt("pad_d");
        pads[1]              = inflags.GetValueInt("pad_h");
        pads[2]              = inflags.GetValueInt("pad_w");
        conv_strides[0]      = inflags.GetValueInt("conv_stride_d");
        conv_strides[1]      = inflags.GetValueInt("conv_stride_h");
        conv_strides[2]      = inflags.GetValueInt("conv_stride_w");
        conv_dilations[0]    = inflags.GetValueInt("dilation_d");
        conv_dilations[1]    = inflags.GetValueInt("dilation_h");
        conv_dilations[2]    = inflags.GetValueInt("dilation_w");
        trans_output_pads[0] = inflags.GetValueInt("trans_output_pad_d");
        trans_output_pads[1] = inflags.GetValueInt("trans_output_pad_h");
        trans_output_pads[2] = inflags.GetValueInt("trans_output_pad_w");
    }
    else
    {
        MIOPEN_THROW("unsupported convolution dimension");
    }

    int out_c       = inflags.GetValueInt("out_channels");
    int in_c        = inflags.GetValueInt("in_channels");
    int group_count = std::max(inflags.GetValueInt("group_count"), 1);

    if(group_count > 1)
    {
        if(in_c % group_count != 0 || out_c % group_count != 0 || group_count > in_c ||
           group_count > out_c)
        {
            printf("Invalid group number\n");
            exit(0); // NOLINT (concurrency-mt-unsafe)
        }
    }

    miopenConvolutionMode_t mode;
    if((inflags.GetValueStr("mode")) == "conv")
    {
        mode = miopenConvolution;
    }
    else if((inflags.GetValueStr("mode")) == "trans")
    {
        mode = miopenTranspose;
    }
    else
    {
        printf("Incorrect Convolution Mode\n");
        exit(0); // NOLINT (concurrency-mt-unsafe)
    }

    // adjust padding based on user-defined padding mode
    if(mode == miopenConvolution &&
       (all_of(conv_dilations, [](auto v) { return v == 1; }) ||
        all_of(wei_spatial_lens, [](auto v) { return v == 1; })))
    {
        if((inflags.GetValueStr("pad_mode")) == "same")
        {
            for(int i = 0; i < spatial_dim; ++i)
            {
                pads[i] =
                    (in_spatial_lens[i] % conv_strides[i] == 0)
                        ? (std::max((wei_spatial_lens[i] - conv_strides[i]), 0))
                        : (std::max((wei_spatial_lens[i] - (in_spatial_lens[i] % conv_strides[i])),
                                    0));
                pads[i] /= 2;
            }
        }
        else if((inflags.GetValueStr("pad_mode")) == "valid")
        {
            for(int i = 0; i < spatial_dim; ++i)
            {
                pads[i] = 0;
            }
        }
    }

    miopenInitConvolutionNdDescriptor(
        convDesc, spatial_dim, pads.data(), conv_strides.data(), conv_dilations.data(), mode);

    miopenSetConvolutionGroupCount(convDesc, group_count);

    if(mode == miopenTranspose)
    {
        miopenSetTransposeConvNdOutputPadding(convDesc, spatial_dim, trans_output_pads.data());
    }

    return miopenStatusSuccess;
}

std::vector<int> GetBiasTensorLengthsFromCmdLine()
{
    int spatial_dim = inflags.GetValueInt("spatial_dim");

    std::vector<int> bias_lens(2 + spatial_dim, 1);

    bias_lens[1] = inflags.GetValueInt("out_channels");

    return bias_lens;
}

inline miopenTensorLayout_t StringToLayoutType(std::string layout)
{
    miopenTensorLayout_t default_layout = miopenTensorNCHW;
    if(layout == "NCHWc4")
        return miopenTensorNCHWc4;
    else if(layout == "NCHWc8")
        return miopenTensorNCHWc8;
    else if(layout == "CHWNc4")
        return miopenTensorCHWNc4;
    else if(layout == "CHWNc8")
        return miopenTensorCHWNc8;
    else
    {
        MIOPEN_THROW("We only support NCHWc4, NCHWc8, CHWNc4, CHWNc8 vectorized tensor layout.");
        return default_layout;
    }
}

inline int SetTensorNd(miopenTensorDescriptor_t t,
                       std::vector<int>& len,
                       miopenDataType_t data_type = miopenFloat)
{
    return miopenSetTensorDescriptor(t, data_type, len.size(), len.data(), nullptr);
}

inline int SetTensorNd(miopenTensorDescriptor_t t,
                       std::vector<int>& len,
                       std::vector<int>& strides,
                       miopenDataType_t data_type = miopenFloat)
{
    return miopenSetTensorDescriptor(t, data_type, len.size(), len.data(), strides.data());
}

inline int SetTensorNd(miopenTensorDescriptor_t t,
                       std::vector<int>& len,
                       const std::string& layout,
                       miopenDataType_t data_type = miopenFloat)
{
    if(layout.empty())
    {
        return SetTensorNd(t, len, data_type);
    }

    if(layout.size() != len.size() && layout.find("c") == std::string::npos)
    {
        MIOPEN_THROW("unmatched layout and dimension size");
    }

    // Dimension lengths vector 'len' comes with a default layout.
    std::string len_layout = tensor_layout_get_default(layout.size());
    if(len_layout.empty())
    {
        return SetTensorNd(t, len, data_type);
    }

    std::vector<int> strides;
    tensor_layout_to_strides(len, len_layout, layout, strides);

    return SetTensorNd(t, len, strides, data_type);
}

std::vector<int> GetOutputTensorLengths()
{
    int ndim = miopen::deref(inputTensor).GetSize();

    std::vector<int> out_lens(ndim);

    // FIXME(iwooook)
    // miopenGetConvolutionNdForwardOutputDim(
    //    convDesc, inputTensor, weightTensor, &ndim, out_lens.data());

    out_lens[0] = key_in_channels;
    out_lens[1] = key_out_channels;
    out_lens[2] = (ndim == 4) ? key_out_lens[1] : key_out_lens[0];
    out_lens[3] = (ndim == 4) ? key_out_lens[2] : key_out_lens[1];
    if(ndim == 5)
        out_lens[4] = key_out_lens[2];

    return out_lens;
}

int GetandSetData()
{
    std::vector<int> in_len  = GetInputTensorLengthsFromCmdLine();
    std::vector<int> wei_len = GetWeightTensorLengthsFromCmdLine();

    SetTensorNd(inputTensor, in_len, inflags.GetValueStr("in_layout"), data_type);
    SetTensorNd(weightTensor, wei_len, inflags.GetValueStr("fil_layout"), data_type);

    SetConvDescriptorFromCmdLineArgs();

    std::vector<int> out_len = GetOutputTensorLengths();
    miopenDataType_t y_type =
        (data_type == miopenInt8 || data_type == miopenInt8x4) ? miopenInt32 : data_type;
    SetTensorNd(outputTensor, out_len, inflags.GetValueStr("out_layout"), y_type);

    if(inflags.GetValueInt("bias") != 0)
    {
        std::vector<int> bias_len = GetBiasTensorLengthsFromCmdLine();
        SetTensorNd(biasTensor, bias_len, data_type);
    }

    return 0;
}

int main()
{
    miopenCreateTensorDescriptor(&inputTensor);
    miopenCreateTensorDescriptor(&weightTensor);
    miopenCreateTensorDescriptor(&outputTensor);
    miopenCreateTensorDescriptor(&biasTensor);

    miopenCreateConvolutionDescriptor(&convDesc);

    // FIXME
    // std::string filename = "/data/MIOpen_moreh/MIOpen/src/kernels/gfx90a68.HIP.fdb.txt";
    std::string in_filename = "/data/MIOpen_moreh/MIOpen/build_ocl/test.fdb.txt";
#if USE_CUSTOM_API
    std::string out_filename = "/data/MIOpen_moreh/MIOpen/build_ocl/maf.db.custom.txt";
#else
    std::string out_filename = "/data/MIOpen_moreh/MIOpen/build_ocl/maf.db.txt";
#endif

    std::ifstream in_file(in_filename);
    std::ofstream out_file(out_filename);

    if(!in_file)
    {
        std::cerr << "find-db file not readable: " << in_filename << std::endl;
        return 0;
    }

    if(!out_file)
    {
        std::cerr << "maf-db file not writeable: " << out_filename << std::endl;
        return 0;
    }

    int n_line = 0;
    while(true)
    {
        std::string line;
        if(!std::getline(in_file, line))
            break;
        ++n_line;

        const auto key_size = line.find('=');
        const bool is_key   = (key_size != std::string::npos && key_size != 0);
        if(!is_key)
        {
            if(!line.empty())
            {
                std::cerr << "Ill-formed record: key not found: " << in_filename << "#" << n_line
                          << std::endl;
            }
            continue;
        }
        const auto current_key = line.substr(0, key_size);

        // init desc
        AddCmdLineArgs();
        // set desc
        ParseKey(current_key);
        // generate desc
        GetandSetData();

        size_t solutionCount;
        miopenConvSolution_t solution;
        bool fallbackPathTaken;

        TensorDescriptor inputTensor_   = deref(inputTensor);
        TensorDescriptor weightTensor_  = deref(weightTensor);
        TensorDescriptor outputTensor_  = deref(outputTensor);
        ConvolutionDescriptor convDesc_ = deref(convDesc);

        Handle handle;
        ConvolutionContext ctx;

        bool ignoreAsmBuild = true;
        bool ret_kernel = true, ret_program = true;

#if USE_CUSTOM_API
        ConstData_t i1, i2;
        Data_t o1;
        i1 = (ConstData_t)1;
        i2 = (ConstData_t)1;
        o1 = (Data_t)1;

        if(convDirection == 0)
        {
            convDesc_.CheckConvFwdUsePreCompiledKernel(handle,
                                                       inputTensor_,
                                                       i1,
                                                       weightTensor_,
                                                       i2,
                                                       outputTensor_,
                                                       o1,
                                                       ignoreAsmBuild,
                                                       &ret_kernel);
        }
        else if(convDirection == 1)
        {
            convDesc_.CheckConvBwdDataUsePreCompiledKernel(handle,
                                                           inputTensor_,
                                                           i1,
                                                           weightTensor_,
                                                           i2,
                                                           outputTensor_,
                                                           o1,
                                                           ignoreAsmBuild,
                                                           &ret_kernel);
        }
        else if(convDirection == 2)
        {
            convDesc_.CheckConvBwdWeightsUsePreCompiledKernel(handle,
                                                              inputTensor_,
                                                              i1,
                                                              outputTensor_,
                                                              i2,
                                                              weightTensor_,
                                                              o1,
                                                              ignoreAsmBuild,
                                                              &ret_kernel);
        }
        else
        {
            std::cerr << "Invalid convDirection" << std::endl;
            exit(1);
        }
#else
        // get solution
        if(convDirection == 0)
        {
            // F
            // set problem
            ProblemDescription problem(inputTensor_,
                                               weightTensor_,
                                               outputTensor_,
                                               convDesc_,
                                               conv::Direction::Forward);

            // set ctx
            ctx = ConvolutionContext{problem};
            ctx.SetStream(&handle);

            // get solution
            convDesc_.GetForwardSolutions(handle,
                                          weightTensor_,
                                          inputTensor_,
                                          outputTensor_,
                                          1,
                                          &solutionCount,
                                          &solution,
                                          &fallbackPathTaken);

            // set new ctx for forward
            ctx = ConvolutionContext{inputTensor_,
                                             weightTensor_,
                                             outputTensor_,
                                             convDesc_,
                                             conv::Direction::Forward};
            ctx.SetStream(&handle);
        }
        else if(convDirection == 1)
        {
            // B
            // set problem
            ProblemDescription problem(outputTensor_,
                                               weightTensor_,
                                               inputTensor_,
                                               convDesc_,
                                               conv::Direction::BackwardData);

            // set ctx
            ctx = ConvolutionContext{problem};
            ctx.SetStream(&handle);

            // get solution
            convDesc_.GetBackwardSolutions(handle,
                                           inputTensor_,
                                           weightTensor_,
                                           outputTensor_,
                                           1,
                                           &solutionCount,
                                           &solution,
                                           &fallbackPathTaken);
        }
        else if(convDirection == 2)
        {
            // W
            // set problem
            ProblemDescription problem(outputTensor_,
                                               weightTensor_,
                                               inputTensor_,
                                               convDesc_,
                                               conv::Direction::BackwardWeights);

            // set ctx
            ctx = ConvolutionContext{problem};
            ctx.SetStream(&handle);

            // get solution
            convDesc_.GetWrwSolutions(handle,
                                      inputTensor_,
                                      outputTensor_,
                                      weightTensor_,
                                      1,
                                      &solutionCount,
                                      &solution,
                                      &fallbackPathTaken);
        }
        else
        {
            std::cerr << "Invalid convDirection" << std::endl;
            exit(1);
        }

        // get solver
        auto solver_id = solver::Id(solution.solution_id);

        FindDbRecord fdb_record{handle, ctx};

        ret_kernel = true;
        for(const auto& pair : fdb_record)
        {
            if(solver::Id{pair.second.solver_id} != solver_id)
                continue;

            const auto&& kernels = handle.GetKernels(pair.second.kcache_key.algorithm_name,
                                                     pair.second.kcache_key.network_config);

            if(!kernels.empty())
                continue;

            auto solver   = solver_id.GetSolver();
            auto db       = GetDb(ctx);
            auto solution = solver.FindSolution(ctx, db, {});

            auto algorithm_name = pair.second.kcache_key.algorithm_name;
            auto network_config = pair.second.kcache_key.network_config;

            if(algorithm_name.empty() || network_config.empty())
            {
                assert(algorithm_name.empty() && network_config.empty());
            }

            ret_program = true;
            for(auto& k : solution.construction_params)
            {
                if(ignoreAsmBuild && boost::algorithm::ends_with(k.kernel_file, ".s"))
                {
                    MIOPEN_LOG_I2(
                        "Passing because ignoreAsmBuild=1, kernel_name = " << k.kernel_name);
                    continue;
                }
                bool has_pre_compiled_program =
                    handle.HasPreCompiledProgram(k.kernel_file, k.comp_options);

                MIOPEN_LOG_I2("has_pre_compiled_program = " << has_pre_compiled_program
                                                            << ", kernel_name = " << k.kernel_name);
                ret_program = ret_program && has_pre_compiled_program;
            }
            ret_kernel = ret_kernel && ret_program;
        }
#endif
        if(ret_kernel)
            out_file << current_key << std::endl;
    }

    miopenDestroyTensorDescriptor(biasTensor);
    miopenDestroyTensorDescriptor(outputTensor);
    miopenDestroyTensorDescriptor(weightTensor);
    miopenDestroyTensorDescriptor(inputTensor);

    miopenDestroyConvolutionDescriptor(convDesc);
}