/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#include <cassert>
#include <miopen/fusion.hpp>
#include <miopen/md_graph.hpp>
#include <miopen/fusion_plan.hpp>
#include <miopen/logger.hpp>
#include <miopen/handle.hpp>
#include <miopen/visit_float.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/solver_id.hpp>
#include <miopen/fusion/solvers.hpp>
#include <miopen/fusion/fusion_invoke_params.hpp>
#include <miopen/solver_id.hpp>

#include <ostream>
#include <ios>
#include <algorithm>
#include <string>
#include <half.hpp>

namespace miopen {

miopenStatus_t ConvBiasActivFusion(Handle& handle,
                                   const void* alpha1,
                                   const TensorDescriptor& xDesc,
                                   ConstData_t x,
                                   const TensorDescriptor& wDesc,
                                   ConstData_t w,
                                   const ConvolutionDescriptor& conv_desc,
                                   miopenConvFwdAlgorithm_t algo,
                                   void* workspace,
                                   size_t workspaceSizeInBytes,
                                   const void* alpha2,
                                   const TensorDescriptor& zDesc,
                                   ConstData_t z,
                                   const TensorDescriptor& biasDesc,
                                   ConstData_t bias,
                                   const ActivationDescriptor& activationDesc,
                                   const TensorDescriptor& yDesc,
                                   Data_t y)
{
    assert(workspace == nullptr);
    assert(workspaceSizeInBytes == 0);
    std::ignore = workspace;
    std::ignore = workspaceSizeInBytes;
    if(alpha1 != nullptr)
    {
        const auto falpha1 = *(static_cast<const float*>(alpha1));
        if(falpha1 != 1.0f)
            MIOPEN_THROW(miopenStatusNotImplemented, "alpha1 can only be 1.0");
    }
    if(alpha2 != nullptr)
    {
        const auto falpha2 = *(static_cast<const float*>(alpha2));
        if(falpha2 != 1.0f)
            MIOPEN_THROW(miopenStatusNotImplemented, "alpha2 can only be 1.0");
    }
    if(z != nullptr || zDesc.GetSize() != 0)
        MIOPEN_THROW(miopenStatusNotImplemented, "The addition of z vector is not yet supported");
    FusionPlanDescriptor fusePlanDesc{miopenVerticalFusion, xDesc};
    OperatorArgs fusionArgs;
    auto convoOp = std::make_shared<ConvForwardOpDescriptor>(conv_desc, wDesc);
    auto biasOp  = std::make_shared<BiasFusionOpDescriptor>(biasDesc);
    auto activOp = std::make_shared<ActivFwdFusionOpDescriptor>(activationDesc.GetMode());
    MIOPEN_CHECK(fusePlanDesc.AddOp(convoOp));
    MIOPEN_CHECK(fusePlanDesc.SetConvAlgo(algo));
    MIOPEN_CHECK(fusePlanDesc.AddOp(biasOp));
    MIOPEN_CHECK(fusePlanDesc.AddOp(activOp));

    MIOPEN_CHECK(fusePlanDesc.Compile(handle));
    float alpha       = static_cast<float>(1.0);
    float beta        = static_cast<float>(0);
    float activ_alpha = activationDesc.GetAlpha();
    float activ_beta  = activationDesc.GetBeta();
    float activ_gamma = activationDesc.GetGamma();

    // Set the Args
    MIOPEN_CHECK(convoOp->SetArgs(&alpha, &beta, w));
    MIOPEN_CHECK(activOp->SetArgs(&alpha, &beta, activ_alpha, activ_beta, activ_gamma));
    MIOPEN_CHECK(biasOp->SetArgs(&alpha, &beta, bias));
    MIOPEN_CHECK(fusePlanDesc.Execute(handle, xDesc, x, yDesc, y, fusionArgs));
    return miopenStatusSuccess;
}

FusionPlanDescriptor::FusionPlanDescriptor(const miopenFusionDirection_t dir,
                                           const TensorDescriptor& inDesc)
    : fusion_dir(dir),
      input_desc(inDesc),
      is_valid(false),
      kernel_source_type(OpenclText),
      fp_contains_bn(false),
      data_type(inDesc.GetType())
{
}

FusionPlanDescriptor::~FusionPlanDescriptor() { op_map.clear(); }

miopenStatus_t FusionPlanDescriptor::AddOp(std::shared_ptr<FusionOpDescriptor> desc)
{
#if 0
    // load the md graph for the first op
    if(op_count == 0)
    {
        FusionMDGraph::Init(lu, desc->kind());
    }
#endif
    desc->SetIdx(op_count);
    if(op_map.empty())
        desc->SetInputDesc(input_desc);
    else
        desc->SetInputDesc(output_desc);
    desc->GetOutputDesc(output_desc);
    op_map.emplace_back(desc);
    op_count++;
#if 0
    is_valid = false;
    miopen::try_([&] {
        is_valid = lu.Advance(desc, [&](const std::string& sym, int& val) -> bool {
            // check tensor attr
            if(GetTensorAttr(sym, val))
                return true;
            // check op attr
            if(desc->GetOpAttr(sym, val))
                return true;
            // check the values of enum types
            if(GetEnumVal(sym, val))
                return true;
            // check dev attr
            // if(GetDevAttribute(sym, val, handle))
            //     return true;
            return false;
        });
    });
    if(is_valid)
        return miopenStatusSuccess;
    else
        return miopenStatusUnsupportedOp;
#endif
    return miopenStatusSuccess;
}

miopenStatus_t FusionPlanDescriptor::GetOp(int op_idx, std::shared_ptr<FusionOpDescriptor>& desc)
{
    auto err = miopenStatusSuccess;

    if(op_idx >= op_map.size())
    {
        MIOPEN_THROW(miopenStatusBadParm, "Operator index out of bounds");
    }

    desc = op_map.at(op_idx);
    return err;
}

TensorDescriptor FusionPlanDescriptor::FusionPlanDescriptor::DeriveOutputDescriptor()
{
    TensorDescriptor i_desc = input_desc;
    TensorDescriptor o_desc;
    if(fusion_dir == miopenVerticalFusion)
    {
        // All the ops should have the same output descriptor otherwise
        // fusion would not be feasible, thus we need to call GetOutputDesc on all
        // the ops and make sure it returns the same value
        for(auto&& op : op_map)
        {
            op->SetInputDesc(i_desc);
            op->GetOutputDesc(o_desc);
            i_desc = o_desc;
        }
    }
    else
    {
        MIOPEN_THROW(miopenStatusNotImplemented, "Unsupported fusion direction");
    }
    return o_desc;
}

miopenStatus_t FusionPlanDescriptor::GetWorkspaceSizeImmed(Handle& handle,
                                                           size_t& workSpaceSize,
                                                           miopenConvFwdAlgorithm_t /*algo*/)
{
    workSpaceSize = 0;
    for(auto&& op : op_map)
    {
        if(op->kind() == miopenFusionOpConvForward)
        {
            auto ptr = std::dynamic_pointer_cast<ConvForwardOpDescriptor>(op);
            TensorDescriptor opd;
            ptr->GetOutputDesc(opd);
            size_t tmp_sz = ptr->base_desc.ForwardGetWorkSpaceSize(
                handle, ptr->filter_desc, ptr->input_desc, opd);
            if(tmp_sz > workSpaceSize)
                workSpaceSize = tmp_sz;
        }
    }
    return miopenStatusSuccess;
}

miopenStatus_t FusionPlanDescriptor::GetConvAlgos(int /*reqAlgoCount*/,
                                                  int& /*retAlgoCount*/,
                                                  miopenConvFwdAlgorithm_t* /*ptrAlgos*/)
{
#if 0
    auto algos   = lu.GetConvAlgos();
    retAlgoCount = std::min(reqAlgoCount, static_cast<int>(algos.size()));

    for(auto idx = 0; idx < retAlgoCount; idx++)
    {
        ptrAlgos[idx] = algos[idx];
    }
    return miopenStatusNotImplemented;
#else
    return miopenStatusNotImplemented;
#endif
}

miopenStatus_t FusionPlanDescriptor::SetConvAlgo(miopenConvFwdAlgorithm_t /*algo*/)
{
#if 0
    bool res = lu.SetConvAlgo(algo);
#else
    bool res = false;
#endif
    if(res)
        return miopenStatusSuccess;
    else
        return miopenStatusUnknownError;
}

std::ostream& operator<<(std::ostream& stream, const FusionPlanDescriptor& /*fpd*/)
{
    // stream << "kernel_name: " << fpd.kernel_name;
    return stream;
}

// Fusion operator descriptors
// Conv Forward
miopenStatus_t ConvForwardOpDescriptor::GetOutputDesc(TensorDescriptor& output_desc) const
{
    return miopen::try_(
        [&]() { output_desc = base_desc.GetForwardOutputTensor(input_desc, filter_desc); });
}

miopenStatus_t
ConvForwardOpDescriptor::SetArgs(const void* /*alpha*/, const void* /*beta*/, ConstData_t w)
{
    const auto& op_args = std::dynamic_pointer_cast<fusion::ConvolutionOpInvokeParam>(args);
    op_args->weights    = w;
    return miopenStatusSuccess;
}

std::shared_ptr<fusion::FusionOpInvokeParamBase> ConvForwardOpDescriptor::GetArgs() const
{
    return args;
}

std::string ConvForwardOpDescriptor::GetArgKey(const std::string& k) const
{
    return k + std::to_string(GetIdx());
}

bool ConvForwardOpDescriptor::GetOpAttr(const std::string& sym, int& val) const
{
    int o, c, x, y;
    std::tie(o, c, x, y) = tien<4>(filter_desc.GetLengths());

    auto f_strides     = filter_desc.GetStrides();
    const int f_t_size = miopen::GetTypeSize(input_desc.GetType());
    std::transform(f_strides.begin(),
                   f_strides.end(),
                   f_strides.begin(),
                   [&f_t_size](const auto& s) { return s * f_t_size; });

    if(sym == "x")
    {
        val = x;
    }
    else if(sym == "y")
    {
        val = y;
    }
    else if(sym == "c")
    {
        val = c;
    }
    else if(sym == "pad_h")
    {
        val = base_desc.GetConvPads()[0];
    }
    else if(sym == "pad_w")
    {
        val = base_desc.GetConvPads()[1];
    }
    else if(sym == "dilation_h")
    {
        val = base_desc.GetConvDilations()[0];
    }
    else if(sym == "dilation_w")
    {
        val = base_desc.GetConvDilations()[1];
    }
    else if(sym == "stride_h")
    {
        val = base_desc.GetConvStrides()[0];
    }
    else if(sym == "stride_w")
    {
        val = base_desc.GetConvStrides()[1];
    }
    else if(sym == "k")
    {
        val = o;
    }
    else if(sym == "group_count")
    {
        val = base_desc.GetGroupCount();
    }
    else if(sym == "f_byte_stride_nk")
    {
        val = f_strides[0];
    }
    else if(sym == "f_byte_stride_c")
    {
        val = f_strides[1];
    }
    else if(sym == "f_byte_stride_h")
    {
        val = f_strides[2];
    }
    else if(sym == "f_byte_stride_w")
    {
        val = f_strides[3];
    }
    else
        return false;

    return true;
}

OpKernelArg ConvForwardOpDescriptor::GetOpAttr(const std::string& k) const
{
    int v;
    if(GetOpAttr(k, v))
    {
        return {v};
    }
    else
    {
        MIOPEN_THROW(miopenStatusInternalError, "Unknown Convolution Op Attribute");
    }
}
// Activ Forward ------------------------------------

miopenStatus_t ActivFwdFusionOpDescriptor::SetArgs(const void* /*alpha*/,
                                                   const void* /*beta*/,
                                                   double activAlpha,
                                                   double activBeta,
                                                   double activGamma)
{
    const auto& op_args = std::dynamic_pointer_cast<fusion::ActivationOpInvokeParam>(args);
    op_args->activAlpha = activAlpha;
    op_args->activBeta  = activBeta;
    op_args->activGamma = activGamma;
#if 0
    auto id = std::to_string(GetIdx());
    if(input_desc.GetType() == miopenFloat)
    {
        args.ins_arg("activAlpha" + id, OpKernelArg(static_cast<float>(activAlpha)));
        args.ins_arg("activBeta" + id, OpKernelArg(static_cast<float>(activBeta)));
        args.ins_arg("activGamma" + id, OpKernelArg(static_cast<float>(activGamma)));
    }
    else if(input_desc.GetType() == miopenHalf)
    {
        args.ins_arg("activAlpha" + id,
                     OpKernelArg(static_cast<half_float::half>(
                         activAlpha))); // NOLINT (cppcoreguidelines-narrowing-conversions)
        args.ins_arg("activBeta" + id,
                     OpKernelArg(static_cast<half_float::half>(
                         activBeta))); // NOLINT (cppcoreguidelines-narrowing-conversions)
        args.ins_arg("activGamma" + id,
                     OpKernelArg(static_cast<half_float::half>(
                         activGamma))); // NOLINT (cppcoreguidelines-narrowing-conversions)
    }
#endif
    return miopenStatusSuccess;
}

std::string ActivFwdFusionOpDescriptor::GetArgKey(const std::string& k) const
{
    return k + std::to_string(GetIdx());
}

std::shared_ptr<fusion::FusionOpInvokeParamBase> ActivFwdFusionOpDescriptor::GetArgs() const
{
    return args;
#if 0
    std::shared_ptr<fusion::FusionOpInvokeParamBase> keys;
    auto id = std::to_string(GetIdx());
    if(input_desc.GetType() == miopenFloat)
    {
        float a = 0.0;
        keys.emplace_back("activAlpha" + id, OpKernelArg(a));
        keys.emplace_back("activBeta" + id, OpKernelArg(a));
        keys.emplace_back("activGamma" + id, OpKernelArg(a));
    }
    else if(input_desc.GetType() == miopenHalf)
    {
        half_float::half a;
        keys.emplace_back("activAlpha" + id, OpKernelArg(a));
        keys.emplace_back("activBeta" + id, OpKernelArg(a));
        keys.emplace_back("activGamma" + id, OpKernelArg(a));
    }

    return keys;
#endif
}

bool ActivFwdFusionOpDescriptor::GetOpAttr(const std::string& sym, int& val) const
{
    if(sym == "activ_mode")
    {
        val = activMode;
        return true;
    }
    return false;
}

OpKernelArg ActivFwdFusionOpDescriptor::GetOpAttr(const std::string& k) const
{
    int v;
    if(GetOpAttr(k, v))
    {
        return {v};
    }
    MIOPEN_THROW(miopenStatusInternalError, "Unknown Activation Op Attribute");
}

miopenStatus_t ActivFwdFusionOpDescriptor::GetOutputDesc(TensorDescriptor& output_desc) const
{
    // activation does not change the size
    output_desc = input_desc;
    return miopenStatusSuccess;
}
// Activ Backwards-----------------------------------------
miopenStatus_t ActivBwdFusionOpDescriptor::SetArgs(const void* /*alpha*/,
                                                   const void* /*beta*/,
                                                   const void* y,
                                                   const void* x,
                                                   double activAlpha,
                                                   double activBeta,
                                                   double activGamma)
{
    const auto& op_args = std::dynamic_pointer_cast<fusion::ActivationBwdOpInvokeParam>(args);
    op_args->y          = y;
    op_args->x          = x;
    op_args->activAlpha = activAlpha;
    op_args->activBeta  = activBeta;
    op_args->activGamma = activGamma;
    return miopenStatusSuccess;
#if 0
    auto id             = std::to_string(GetIdx());
    auto activDiffScale = activBeta * activGamma;
    if(input_desc.GetType() == miopenFloat)
    {
        args.ins_arg("activAlpha" + id, OpKernelArg(static_cast<float>(activAlpha)));
        args.ins_arg("activBeta" + id, OpKernelArg(static_cast<float>(activBeta)));
        args.ins_arg("activGamma" + id, OpKernelArg(static_cast<float>(activGamma)));
        args.ins_arg("activDiffScale" + id, OpKernelArg(static_cast<float>(activDiffScale)));
    }
    else if(input_desc.GetType() == miopenHalf)
    {
        args.ins_arg("activAlpha" + id,
                     OpKernelArg(static_cast<half_float::half>(
                         activAlpha))); // NOLINT (cppcoreguidelines-narrowing-conversions)
        args.ins_arg("activBeta" + id,
                     OpKernelArg(static_cast<half_float::half>(
                         activBeta))); // NOLINT (cppcoreguidelines-narrowing-conversions)
        args.ins_arg("activGamma" + id,
                     OpKernelArg(static_cast<half_float::half>(
                         activGamma))); // NOLINT (cppcoreguidelines-narrowing-conversions)
        args.ins_arg("activDiffScale" + id,
                     OpKernelArg(static_cast<half_float::half>(activDiffScale)));
    }

    auto y_any = OpKernelArg(y);
    auto x_any = OpKernelArg(x);
    args.ins_arg("y" + id, y_any);
    args.ins_arg("x" + id, x_any);
#endif
}

std::string ActivBwdFusionOpDescriptor::GetArgKey(const std::string& k) const
{
    return k + std::to_string(GetIdx());
}

OpKernelArg ActivBwdFusionOpDescriptor::GetOpAttr(const std::string& k) const
{
    MIOPEN_THROW("ActivBwdFusionOpDescriptor op does not support attribute: " + k);
}

std::shared_ptr<fusion::FusionOpInvokeParamBase> ActivBwdFusionOpDescriptor::GetArgs() const
{
    return args;
#if 0
    fusion::FusionOpInvokeParamBase keys;
    auto id = std::to_string(GetIdx());
    if(input_desc.GetType() == miopenFloat)
    {
        float a = 0.0;
        keys.emplace_back("activAlpha" + id, OpKernelArg(a));
        keys.emplace_back("activBeta" + id, OpKernelArg(a));
        keys.emplace_back("activGamma" + id, OpKernelArg(a));
    }
    else if(input_desc.GetType() == miopenHalf)
    {
        half_float::half a;
        keys.emplace_back("activAlpha" + id, OpKernelArg(a));
        keys.emplace_back("activBeta" + id, OpKernelArg(a));
        keys.emplace_back("activGamma" + id, OpKernelArg(a));
    }
    keys.emplace_back("activDiffScale" + id, OpKernelArg(nullptr));
    keys.emplace_back("y" + id, OpKernelArg(nullptr));
    keys.emplace_back("x" + id, OpKernelArg(nullptr));
    return keys;
#endif
}

miopenStatus_t ActivBwdFusionOpDescriptor::GetOutputDesc(TensorDescriptor& output_desc) const
{
    // activation does not change the size
    output_desc = input_desc;
    return miopenStatusSuccess;
}
//==============================

miopenStatus_t BatchNormInferenceFusionOpDescriptor::SetArgs(const void*,
                                                             const void*,
                                                             ConstData_t bnScale,
                                                             ConstData_t bnBias,
                                                             ConstData_t estimatedMean,
                                                             ConstData_t estimatedVariance,
                                                             double epsilon)
{
    const auto& op_args = std::dynamic_pointer_cast<fusion::BatchNormInferenceOpInvokeParam>(args);
    op_args->bnScale    = bnScale;
    op_args->bnBias     = bnBias;
    op_args->estimatedMean     = estimatedMean;
    op_args->estimatedVariance = estimatedVariance;
    op_args->epsilon           = epsilon;
#if 0
    auto id                    = std::to_string(GetIdx());
    auto bnScale_any           = OpKernelArg(bnScale);
    auto bnBias_any            = OpKernelArg(bnBias);
    auto estimatedMean_any     = OpKernelArg(estimatedMean);
    auto estimatedVariance_any = OpKernelArg(estimatedVariance);
    auto epsilon_any           = OpKernelArg(static_cast<double>(epsilon));
    args.ins_arg("epsilon" + id, epsilon_any);
    args.ins_arg("bnScale" + id, bnScale_any);
    args.ins_arg("bnBias" + id, bnBias_any);
    args.ins_arg("estimatedMean" + id, estimatedMean_any);
    args.ins_arg("estimatedVariance" + id, estimatedVariance_any);
#endif
    return miopenStatusSuccess;
}

std::string BatchNormInferenceFusionOpDescriptor::GetArgKey(const std::string& k) const
{
    return k + std::to_string(GetIdx());
}

std::shared_ptr<fusion::FusionOpInvokeParamBase>
BatchNormInferenceFusionOpDescriptor::GetArgs() const
{
    return args;
#if 0
    fusion::FusionOpInvokeParamBase keys;
    auto id        = std::to_string(GetIdx());
    double epsilon = 0.0;
    keys.emplace_back("epsilon" + id, OpKernelArg(epsilon));
    ConstData_t bnScale = nullptr;
    keys.emplace_back("bnScale" + id, OpKernelArg(bnScale));
    ConstData_t bnBias = nullptr;
    keys.emplace_back("bnBias" + id, OpKernelArg(bnBias));
    ConstData_t estimatedMean = nullptr;
    keys.emplace_back("estimatedMean" + id, OpKernelArg(estimatedMean));
    ConstData_t estimatedVariance = nullptr;
    keys.emplace_back("estimatedVariance" + id, OpKernelArg(estimatedVariance));
    return keys;
#endif
}

miopenStatus_t
BatchNormInferenceFusionOpDescriptor::GetOutputDesc(TensorDescriptor& output_desc) const
{
    output_desc = input_desc;
    return miopenStatusSuccess;
}

OpKernelArg BatchNormInferenceFusionOpDescriptor::GetOpAttr(const std::string& k) const
{
    int v;
    if(GetOpAttr(k, v))
    {
        return {v};
    }
    else
    {
        MIOPEN_THROW(miopenStatusInternalError, "Unknown Activation Op Attribute");
    }
}
bool BatchNormInferenceFusionOpDescriptor::GetOpAttr(const std::string& sym, int& val) const
{
    if(sym == "bn_mode")
    {
        val = mode;
        return true;
    }
    else
    {
        return false;
    }
}

// Batch Normalization Forward Training --------------
miopenStatus_t BatchNormFwdTrainFusionOpDescriptor::SetArgs(const void* /*alpha*/,
                                                            const void* /*beta*/,
                                                            Data_t runningMean,
                                                            Data_t runningVariance,
                                                            Data_t savedMean,
                                                            Data_t savedInvVariance,
                                                            ConstData_t bnScale,
                                                            ConstData_t bnBias,
                                                            double expAvgFactor,
                                                            double epsilon)
{
    if(runningMeanVar && (runningMean == nullptr || runningVariance == nullptr))
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     "Save batch statistics was turned on at op creation time "
                     "but runningMean or runningVariance is set to nullptr");
    }

    const auto& op_args =
        std::dynamic_pointer_cast<fusion::BatchNormFwdTrainingOpInvokeParam>(args);
    op_args->runningMean      = runningMean;
    op_args->runningVariance  = runningVariance;
    op_args->savedMean        = savedMean;
    op_args->savedInvVariance = savedInvVariance;
    op_args->bnScale          = bnScale;
    op_args->bnBias           = bnBias;
    op_args->expAvgFactor     = expAvgFactor;
    op_args->epsilon          = epsilon;
    return miopenStatusSuccess;
#if 0
    // @todo add in saved versus running boolean toggles
    auto id                   = std::to_string(GetIdx());
    auto bnScale_any          = OpKernelArg(bnScale);
    auto bnBias_any           = OpKernelArg(bnBias);
    auto runningMean_any      = OpKernelArg(runningMean);
    auto runningVariance_any  = OpKernelArg(runningVariance);
    auto savedMean_any        = OpKernelArg(savedMean);
    auto savedInvVariance_any = OpKernelArg(savedInvVariance);
    auto expAvgFactor_any     = OpKernelArg(static_cast<double>(expAvgFactor));
    auto epsilon_any          = OpKernelArg(static_cast<double>(epsilon));
    int n, c, h, w;
    std::tie(n, c, h, w) = tien<4>(input_desc.GetLengths());
    auto nhw             = static_cast<float>(n * h * w);

    auto inhw_any = static_cast<float>(1.0f / nhw);


    args.ins_arg("inhw" + id, inhw_any);
    args.ins_arg("expAvgFactor" + id, expAvgFactor_any);
    args.ins_arg("epsilon" + id, epsilon_any);
    args.ins_arg("bnScale" + id, bnScale_any);
    args.ins_arg("bnBias" + id, bnBias_any);
    args.ins_arg("savedMean" + id, savedMean_any);
    args.ins_arg("savedInvVariance" + id, savedInvVariance_any);
    args.ins_arg("runningMean" + id, runningMean_any);
    args.ins_arg("runningVariance" + id, runningVariance_any);
#endif
}

std::shared_ptr<fusion::FusionOpInvokeParamBase>
BatchNormFwdTrainFusionOpDescriptor::GetArgs() const
{
    return args;

#if 0
    // @todo add in saved versus running boolean toggles
    fusion::FusionOpInvokeParamBase keys;
    auto id        = std::to_string(GetIdx());
    Data_t d       = nullptr;
    ConstData_t cd = nullptr;
    auto f_any     = OpKernelArg(static_cast<float>(0.0f));
    auto d_any     = OpKernelArg(d);
    auto cd_any    = OpKernelArg(cd);

    if(mode == miopenBNSpatial)
    {
        keys.emplace_back("inhw" + id, f_any);
    }

    keys.emplace_back("epsilon" + id, OpKernelArg(static_cast<double>(0)));
    keys.emplace_back("bnScale" + id, cd_any);
    keys.emplace_back("bnBias" + id, cd_any);
    keys.emplace_back("savedMean" + id, d_any);
    keys.emplace_back("savedInvVariance" + id, d_any);
    keys.emplace_back("expAvgFactor" + id, OpKernelArg(static_cast<double>(0)));
    keys.emplace_back("runningMean" + id, d_any);
    keys.emplace_back("runningVariance" + id, d_any);
    return keys;
#endif
}

miopenStatus_t
BatchNormFwdTrainFusionOpDescriptor::GetOutputDesc(TensorDescriptor& output_desc) const
{
    output_desc = input_desc;
    return miopenStatusSuccess;
}

// end BN forward training -----------------------------

// Batch Normalization Backward Training --------------
std::string BatchNormBwdTrainFusionOpDescriptor::GetArgKey(const std::string& k) const
{
    return k + std::to_string(GetIdx());
}
bool BatchNormBwdTrainFusionOpDescriptor::GetOpAttr(const std::string& sym, int& val) const
{
    if(sym == "bn_mode")
    {
        val = mode;
        return true;
    }
    else
    {
        return false;
    }
}
OpKernelArg BatchNormBwdTrainFusionOpDescriptor::GetOpAttr(const std::string& k) const
{
    int v;
    if(GetOpAttr(k, v))
    {
        return {v};
    }
    else if(k == "diff_scale")
    {
        return {static_cast<float>(0.0)};
    }
    else if(k == "iNHW")
    {
        int n, h, w;
        std::tie(n, std::ignore, h, w) = tien<4>(input_desc.GetLengths());
        auto nhw                       = static_cast<float>(n * h * w);
        return {static_cast<float>(1.0f / nhw)};
    }
    else
        MIOPEN_THROW("BatchNormBwdTrainFusionOpDescriptor does not support attribute: " + k);
}
miopenStatus_t BatchNormBwdTrainFusionOpDescriptor::SetArgs(const void* /*alpha*/,
                                                            const void* /*beta*/,
                                                            ConstData_t x,
                                                            ConstData_t bnScale,
                                                            ConstData_t bnBias,
                                                            Data_t resBnScaleDiff,
                                                            Data_t resBnBiasDiff,
                                                            ConstData_t savedMean,
                                                            ConstData_t savedInvVariance)
{
    const auto& op_args =
        std::dynamic_pointer_cast<fusion::BatchNormBwdTrainingOpInvokeParam>(args);

    op_args->x                = x;
    op_args->bnScale          = bnScale;
    op_args->bnBias           = bnBias;
    op_args->resBnScaleDiff   = resBnScaleDiff;
    op_args->resBnBiasDiff    = resBnBiasDiff;
    op_args->savedMean        = savedMean;
    op_args->savedInvVariance = savedInvVariance;
    return miopenStatusSuccess;
#if 0
    // @todo add in saved boolean toggle
    auto id                   = std::to_string(GetIdx());
    auto x_any                = OpKernelArg(x);
    auto bnScale_any          = OpKernelArg(bnScale);
    auto bnBias_any           = OpKernelArg(bnBias);
    auto resBnScaleDiff_any   = OpKernelArg(resBnScaleDiff);
    auto resBnBiasDiff_any    = OpKernelArg(resBnBiasDiff);
    auto savedMean_any        = OpKernelArg(savedMean);
    auto savedInvVariance_any = OpKernelArg(savedInvVariance);

    args.ins_arg("x" + id, x_any);
    args.ins_arg("bnScale" + id, bnScale_any);
    args.ins_arg("bnBias" + id, bnBias_any);
    args.ins_arg("resBnScaleDiff" + id, resBnScaleDiff_any);
    args.ins_arg("resBnBiasDiff" + id, resBnBiasDiff_any);
    args.ins_arg("savedMean" + id, savedMean_any);
    args.ins_arg("savedInvVariance" + id, savedInvVariance_any);
#endif
}

std::shared_ptr<fusion::FusionOpInvokeParamBase>
BatchNormBwdTrainFusionOpDescriptor::GetArgs() const
{
    return args;
#if 0
    fusion::FusionOpInvokeParamBase keys;
    auto id        = std::to_string(GetIdx());
    Data_t d       = nullptr;
    ConstData_t cd = nullptr;
    auto d_any     = OpKernelArg(d);
    auto cd_any    = OpKernelArg(cd);

    keys.emplace_back("x" + id, cd_any);
    keys.emplace_back("bnScale" + id, cd_any);
    keys.emplace_back("bnBias" + id, cd_any);
    keys.emplace_back("resBnScaleDiff" + id, d_any);
    keys.emplace_back("resBnBiasDiff" + id, d_any);
    keys.emplace_back("savedMean" + id, cd_any);
    keys.emplace_back("savedInvVariance" + id, cd_any);
    return keys;
#endif
}

miopenStatus_t
BatchNormBwdTrainFusionOpDescriptor::GetOutputDesc(TensorDescriptor& output_desc) const
{
    output_desc = input_desc;
    return miopenStatusSuccess;
}

// end BN backwards training ---------------------------

// Bias forward
miopenStatus_t BiasFusionOpDescriptor::GetOutputDesc(TensorDescriptor& output_desc) const
{
    output_desc = input_desc;
    return miopenStatusSuccess;
}

miopenStatus_t
BiasFusionOpDescriptor::SetArgs(const void* /*alpha*/, const void* /*beta*/, ConstData_t bdata)
{
    const auto& op_args = std::dynamic_pointer_cast<fusion::BiasOpInvokeParam>(args);
    op_args->bdata      = bdata;
    return miopenStatusSuccess;
#if 0
    auto bdata_any = OpKernelArg(bdata);
    args.ins_arg("bias" + std::to_string(GetIdx()), bdata_any);
#endif
}

std::string BiasFusionOpDescriptor::GetArgKey(const std::string& k) const
{
    return k + std::to_string(GetIdx());
}

OpKernelArg BiasFusionOpDescriptor::GetOpAttr(const std::string& /* k */) const
{
    MIOPEN_THROW(miopenStatusInternalError, "Unknown Bias Op Attribute");
}

std::shared_ptr<fusion::FusionOpInvokeParamBase> BiasFusionOpDescriptor::GetArgs() const
{
    return args;
#if 0
    ConstData_t bdata = nullptr;
    fusion::FusionOpInvokeParamBase keys;
    keys.emplace_back("bias" + std::to_string(GetIdx()), OpKernelArg(bdata));
    return keys;
#endif
}

#if 0
static inline void
find_replace_first(std::string& s_where, const std::string& s_find, const std::string& s_replace)
{
    const auto pos = s_where.find(s_find);
    if(pos != std::string::npos)
        s_where.replace(pos, s_find.length(), s_replace);
}
#endif

std::string FusionPlanDescriptor::GetKernelName(const Handle& /*handle*/)
{
#if 0
    if(!op_map.empty())
    {
        kernel_name = lu.GetKernelName(handle);
        return kernel_name;
    }
    else
#endif
    MIOPEN_THROW(miopenStatusNotImplemented, "Unsupported starting op in Fusion Plan");
}

std::string FusionPlanDescriptor::GetAlgorithmName(const Handle& /*handle*/)
{
#if 0
    if(!op_map.empty())
    {
        algorithm_name = lu.GetAlgoName(handle);
        return algorithm_name;
    }
    else
#endif
    MIOPEN_THROW(miopenStatusNotImplemented, "Unsupported starting op in Fusion Plan");
}

bool FusionPlanDescriptor::GetEnumVal(const std::string& sym, int& val) const
{
    if(sym == "miopenFloat")
    {
        val = miopenFloat;
        return true;
    }
    else if(sym == "miopenConvolutionFwdAlgoDirect")
    {
        val = miopenConvolutionFwdAlgoDirect;
        return true;
    }
    else if(sym == "miopenConvolutionFwdAlgoWinograd")
    {
        val = miopenConvolutionFwdAlgoWinograd;
        return true;
    }
    else if(sym == "miopenBNPerActivation")
    {
        val = miopenBNPerActivation;
        return true;
    }
    else if(sym == "miopenBNSpatial")
    {
        val = miopenBNSpatial;
        return true;
    }
    else if(sym == "miopenActivationRELU")
    {
        val = miopenActivationRELU;
        return true;
    }
    else if(sym == "miopenActivationLEAKYRELU")
    {
        val = miopenActivationLEAKYRELU;
        return true;
    }
    return false;
}

bool FusionPlanDescriptor::GetTensorAttr(const std::string& sym, int& val) const
{
    int N, C, H, W, oN, K, oH, oW;
    std::tie(N, C, H, W)    = miopen::tien<4>(input_desc.GetLengths(), 1);
    std::tie(oN, K, oH, oW) = miopen::tien<4>(output_desc.GetLengths(), 1);

    const int d_t_size = miopen::GetTypeSize(input_desc.GetType());
    const int o_t_size = miopen::GetTypeSize(output_desc.GetType());
    auto d_strides     = input_desc.GetStrides();
    auto o_strides     = output_desc.GetStrides();
    std::transform(d_strides.begin(),
                   d_strides.end(),
                   d_strides.begin(),
                   [&d_t_size](const auto& s) { return s * d_t_size; });
    std::transform(o_strides.begin(),
                   o_strides.end(),
                   o_strides.begin(),
                   [&o_t_size](const auto& s) { return s * o_t_size; });

    if(sym == "iN")
    {
        val = N;
    }
    else if(sym == "iC")
    {
        val = C;
    }
    else if(sym == "iH")
    {
        val = H;
    }
    else if(sym == "iW")
    {
        val = W;
    }
    else if(sym == "oN")
    {
        val = oN;
    }
    else if(sym == "oK")
    {
        val = K;
    }
    else if(sym == "oH")
    {
        val = oH;
    }
    else if(sym == "oW")
    {
        val = oW;
    }
    else if(sym == "d_byte_stride_nk")
    {
        val = d_strides[0];
    }
    else if(sym == "d_byte_stride_c")
    {
        val = d_strides[1];
    }
    else if(sym == "d_byte_stride_h")
    {
        val = d_strides[2];
    }
    else if(sym == "d_byte_stride_w")
    {
        val = d_strides[3];
    }
    else if(sym == "o_byte_stride_nk")
    {
        val = o_strides[0];
    }
    else if(sym == "o_byte_stride_c")
    {
        val = o_strides[1];
    }
    else if(sym == "o_byte_stride_h")
    {
        val = o_strides[2];
    }
    else if(sym == "o_byte_stride_w")
    {
        val = o_strides[3];
    }
    else if(sym == "precision")
    {
        assert(input_desc.GetType() == output_desc.GetType());
        val = input_desc.GetType();
    }
    else
        return false;

    return true;
}

OpKernelArg FusionPlanDescriptor::GetTensorAttr(const std::string& sym) const
{
    int val;
    if(FusionPlanDescriptor::GetTensorAttr(sym, val))
        return {val};
    else
    {
        MIOPEN_THROW(miopenStatusInternalError, "Unknown Tensor Attribute: " + sym);
    }
}
OpKernelArg FusionPlanDescriptor::GetDevAttribute(const std::string& k, const Handle& handle) const
{
    if(k == "devCUs")
    {
        int num_cus = handle.GetMaxComputeUnits();
        return {num_cus};
    }
    else
    {
        MIOPEN_THROW(miopenStatusInternalError, "Unknown device attribute " + k);
    }
}

miopenStatus_t FusionPlanDescriptor::Compile(Handle& handle)
{
    miopenStatus_t status = miopenStatusUnknownError;
    const auto solvers    = solver::SolverContainer<solver::fusion::ConvBiasActivAsm1x1U,
                                                 solver::fusion::ConvOclDirectFwdFused,
                                                 solver::fusion::ConvBinWinogradRxSFused,
                                                 solver::fusion::ConvBinWinogradRxSf2x3g1Fused,
                                                 solver::fusion::BnFwdInferActivationFused,
                                                 solver::fusion::BnFwdTrgActivationFused,
                                                 solver::fusion::BnBwdTrgActivationFused>{};
    auto fusion_ctx       = FusionContext{this, handle};
    fusion_ctx.DetectRocm();
    const auto sols =
        solvers.SearchForAllSolutions(fusion_ctx, miopen::GetDb(fusion_ctx), AnyInvokeParams{});
    auto net_config =
        input_desc.ToString() + ((input_desc.GetType() == miopenHalf) ? "FP16" : "FP32");
    net_config += output_desc.ToString() + ((input_desc.GetType() == miopenHalf) ? "FP16" : "FP32");

    for(const auto& op : op_map)
    {
        op->GetNetworkConfig(net_config, handle);
    }
    network_config = NetworkConfig{net_config};
    if(sols.empty())
    {
        return miopenStatusUnsupportedOp;
    }
    else
    {
        for(const auto& sol : sols)
        {
            if(!sol.invoker_factory)
                MIOPEN_THROW(miopenStatusInternalError,
                             "Invoker missing from solver " + sol.solver_id);
            const auto invoker = handle.PrepareInvoker(
                *sol.invoker_factory, sol.construction_params); // force compilation
            handle.RegisterInvoker(invoker, network_config, sol.solver_id, {});
            solutions.push_back(sol);
        }
        std::sort(solutions.begin(),
                  solutions.end(),
                  [](const solver::ConvSolution& a, const solver::ConvSolution& b) -> bool {
                      return a.weight < b.weight;
                  });
        status = miopenStatusSuccess;
    }
    return status;
}

miopenStatus_t FusionPlanDescriptor::Execute(const Handle& handle,
                                             const TensorDescriptor& inputDesc,
                                             ConstData_t input,
                                             const TensorDescriptor& outputDesc,
                                             Data_t output,
                                             const OperatorArgs& /*op_args*/)
{
    if(output_desc != outputDesc)
    {
        MIOPEN_THROW(miopenStatusBadParm, "The output descriptors dont match.");
    }
    if(input_desc != inputDesc)
    {
        MIOPEN_THROW(miopenStatusBadParm, "The input descriptors dont match.");
    }
    const auto& solution = solutions[0];
    if(!solution.Succeeded())
    {
        MIOPEN_THROW(miopenStatusBadParm, "The Fusion Plan was not compiled");
    }

    std::vector<std::shared_ptr<fusion::FusionOpInvokeParamBase>> params;
    for(const auto& op : op_map)
        params.push_back(op->GetArgs());
    const auto invoker = handle.GetInvoker(network_config, solver::Id{solution.solver_id}, {});
    const auto plan_params =
        fusion::FusionInvokeParams{params, inputDesc, input, outputDesc, output, false};
    (*invoker)(handle, plan_params);

#if 1
    return miopenStatusSuccess;
#else

    auto ops_head = op_map[0];

    auto&& kernels = handle.GetKernels(algorithm_name, network_config);
    MIOPEN_LOG_I(algorithm_name << ',' << network_config);
    if(kernels.empty())
    {
        MIOPEN_THROW(miopenStatusBadParm, "The FusionPlan was not compiled for execution");
    }
    KernelInvoke kernel = kernels.front();

    std::vector<OpKernelArg> args;
    if(arg_list.empty())
    {
        MIOPEN_THROW("Kernel arguments not setup properly");
    }
    for(auto& arg : arg_list)
    {
        MIOPEN_LOG_I2("Key: " + arg.key);
        switch(arg.type)
        {
        case Input_Ptr: args.emplace_back(OpKernelArg(input)); break;
        case Output_Ptr: args.emplace_back(OpKernelArg(output)); break;
        case Padding: args.emplace_back(OpKernelArg(0, arg.size)); break;
        case Scalar:
        case Pointer: {
            auto it = op_args.args_map.find(arg.key);
            if(it != op_args.args_map.end())
            {
                args.push_back(it->second);
            }
            else
            {
                MIOPEN_THROW(miopenStatusInternalError, "Argument Not Set: " + arg.key);
            }
            break;
        }
        case Default: args.push_back(arg.val); break;
        }
    }
    if(args.empty())
    {
        MIOPEN_THROW("Operator args not populated properly");
    }
    kernel(args);
    return miopenStatusSuccess;
#endif
}

} // namespace miopen
