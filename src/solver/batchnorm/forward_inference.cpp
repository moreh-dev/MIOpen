/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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

#include <miopen/batchnorm/solvers.hpp>

#include <miopen/batchnorm/invoke_params.hpp>
#include <miopen/batchnorm/problem_description.hpp>
#include <miopen/batch_norm.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/visit_float.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/fusion/solvers.hpp>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_BN_FWDINFER_ACTIV_FUSED)

namespace miopen {

namespace solver {

namespace fusion {

bool BnFwdInferActivationFused::IsApplicable(const FusionContext& context) const
{
    const auto& desc = *context.problem.fusion_plan_desc;
    if(desc.op_map.empty())
        MIOPEN_THROW("");
    if(miopen::IsDisabled(MIOPEN_DEBUG_BN_FWDINFER_ACTIV_FUSED{}))
        return false;
    if(desc.op_map.size() != 2)
        return false;
    if(desc.op_map.at(0)->kind() != miopenFusionOpBatchNormInference)
        return false;
    if(desc.op_map.at(1)->kind() != miopenFusionOpActivForward)
        return false;

    return true;
}

ConvSolution BnFwdInferActivationFused::GetSolution(const FusionContext& fusion_ctx) const
{
    const auto problem =
        fusion_ctx.problem.GetBnProblem(0, miopen::batchnorm::Direction::ForwardInference);

    auto result = ConvSolution{miopenStatusSuccess};
    auto kernel = KernelInfo{};

    kernel.kernel_file = "MIOpenBatchNormActivInfer.cl";
    kernel.kernel_name = "MIOpenBatchNormActivInfer";
    const auto mode    = problem.GetMode();
    if(mode == miopenBNSpatial)
    { // SPATIAL kernels
        kernel.kernel_name += "SpatialEst";
    }
    else
    { // PER ACTIVATION
        kernel.kernel_name += "PerActEst";
    }
    kernel.l_wk.push_back(256);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    int n, c, h, w;
    const auto& input_desc = problem.GetXDesc();
    std::tie(n, c, h, w)   = tien<4>(input_desc.GetLengths());

    size_t read_unit = 1;
    size_t read_len  = (mode == miopenBNSpatial) ? h * w : c * h * w;

    if(mode == miopenBNSpatial && input_desc.GetType() != miopenHalf)
    {
        read_unit = (read_len % 4 == 0) ? 4 : (read_len % 2 == 0) ? 2 : 1;
    }
    std::string READ_TYPE = (read_unit == 1) ? "_FLOAT" : "_FLOAT" + std::to_string(read_unit);
    const auto& activ_op =
        dynamic_cast<ActivFwdFusionOpDescriptor&>(*fusion_ctx.problem.fusion_plan_desc->op_map[1]);
    const auto build_params = KernelBuildParameters{
        {"MIO_BN_CHW", static_cast<int>(c * h * w)},
        {"MIO_BN_HW", static_cast<int>(h * w)},
        {"MIO_BN_N", static_cast<int>(n)},
        {"MIO_BN_GRP0", kernel.l_wk[0]},
        {"MIO_BN_GRP1", static_cast<int>(1)},
        {"MIO_BN_GRP2", static_cast<int>(1)},
        {"MIOPEN_READ_UNIT", static_cast<int>(read_unit)},
        {"MIOPEN_READ_TYPE", READ_TYPE},
        {"MIOPEN_YES_ACTIV", static_cast<int>(1)},
        {"MIOPEN_NRN_OP_ID", static_cast<int>(activ_op.activMode)},
        {"MIOPEN_USE_FP16", static_cast<int>(input_desc.GetType() == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(input_desc.GetType() == miopenFloat)}};
    kernel.comp_options = build_params.GenerateFor(kbp::OpenCL{});
    if(problem.GetMode() == miopenBNSpatial)
        kernel.comp_options += " -DSPATIAL_BN";
    else
        kernel.comp_options += " -DPERACT_BN";
    if(input_desc.GetType() == miopenHalf)
        kernel.comp_options += " -DMIOPEN_USE_FPMIX=1";
    size_t xgridsize = read_len / read_unit;
    size_t ygridsize = (mode == miopenBNSpatial) ? size_t(c) : 1;
    size_t zgridsize = 1;

    kernel.g_wk.push_back(xgridsize);
    kernel.g_wk.push_back(ygridsize);
    kernel.g_wk.push_back(zgridsize);

    result.construction_params.push_back(kernel);

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) run_kernel = handle_.Run(kernels.front());
            const auto& invoke_ctx    = raw_params.CastTo<miopen::fusion::FusionInvokeParams>();
            const auto& bot_ocl_buf   = invoke_ctx.in;
            const auto& top_ocl_buf   = invoke_ctx.out;
            const auto& bn_invoke =
                std::dynamic_pointer_cast<miopen::fusion::BatchNormInferenceOpInvokeParam>(
                    invoke_ctx.op_invokers[0]);
            const auto& activ_invoker =
                std::dynamic_pointer_cast<miopen::fusion::ActivationOpInvokeParam>(
                    invoke_ctx.op_invokers[1]);
            const auto activ_alpha = activ_invoker->activAlpha;
            const auto activ_beta  = activ_invoker->activBeta;
            const auto activ_gamma = activ_invoker->activGamma;
            if(input_desc.GetType() == miopenFloat)
            {
                run_kernel(static_cast<float>(activ_alpha),
                           static_cast<float>(activ_beta),
                           static_cast<float>(activ_gamma),
                           bn_invoke->epsilon, // double
                           bot_ocl_buf,
                           top_ocl_buf,
                           bn_invoke->bnBias,
                           bn_invoke->bnScale,
                           bn_invoke->estimatedMean,
                           bn_invoke->estimatedVariance);
            }
            else if(input_desc.GetType() == miopenHalf)
            {
                run_kernel(static_cast<half_float::half>(activ_alpha),
                           static_cast<half_float::half>(activ_beta),
                           static_cast<half_float::half>(activ_gamma),
                           bn_invoke->epsilon, // double
                           bot_ocl_buf,
                           top_ocl_buf,
                           bn_invoke->bnBias,
                           bn_invoke->bnScale,
                           bn_invoke->estimatedMean,
                           bn_invoke->estimatedVariance);
            }
        };
    };

    return result;
}
} // namespace fusion

namespace batchnorm {

bool BnFwdInference::IsApplicable(const ExecutionContext&,
                                  const miopen::batchnorm::ProblemDescription& problem) const
{
    return problem.GetDirection() == miopen::batchnorm::Direction::ForwardInference;
}

ConvSolution BnFwdInference::GetSolution(const ExecutionContext& context,
                                         const miopen::batchnorm::ProblemDescription& problem) const
{
    const auto& handle = context.GetStream();

    bool bfpmixparm = false;
    bool bfp16parm  = false;
    bool bfp32parm  = true;
    if(problem.GetXDesc().GetType() == miopenHalf &&
       problem.GetBnScaleBiasMeanVarDesc().GetType() == miopenHalf)
    {
        bfp16parm = true;
        bfp32parm = false;
    }
    else if(problem.GetXDesc().GetType() == miopenHalf &&
            problem.GetBnScaleBiasMeanVarDesc().GetType() == miopenFloat)
    {
        bfpmixparm = true;
        bfp32parm  = false;
    }

    int n, c, h, w;
    std::tie(n, c, h, w) = tien<4>(problem.GetXDesc().GetLengths());

    unsigned int in_cstride = h * w;

    auto result = ConvSolution{miopenStatusSuccess};

    {
        size_t xlocalsize = 1;
        auto xgridsize    = c;
        size_t ylocalsize = 256;
        size_t ygridsize  = ylocalsize * ((in_cstride + ylocalsize - 1) / ylocalsize);
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenBatchNormFwdInfer"; // build this up
        kernel.kernel_name = "MIOpenBatchNormFwdInfer";
        if(problem.GetMode() == miopenBNSpatial)
        { // SPATIAL kernels
            kernel.kernel_file += "Spatial.cl";
            kernel.kernel_name += "SpatialEst";
        }
        else
        { // PER ACTIVATION
            kernel.kernel_file += "PerAct.cl";
            kernel.kernel_name += "PerActivationEst";
        }

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(bfp16parm)},
            {"MIOPEN_USE_FP32", static_cast<int>(bfp32parm)},
            {"MIOPEN_USE_FPMIX", static_cast<int>(bfpmixparm)},
            {"MIO_BN_GRP0", xlocalsize},
            {"MIO_BN_GRP1", ylocalsize},
            {"MIO_BN_GRP2", zlocalsize},
            {"MIO_BN_GFX103X", (StartsWith(handle.GetDeviceName(), "gfx103") ? "1" : "0")},
        };

        kernel.comp_options = build_params.GenerateFor(kbp::OpenCL{});

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(ylocalsize);
        kernel.l_wk.push_back(zlocalsize);

        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(ygridsize);
        kernel.g_wk.push_back(zgridsize);

        result.construction_params.push_back(kernel);
    }

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::batchnorm::InfInvokeParams>();

            int n_, c_, h_, w_;
            std::tie(n_, c_, h_, w_) = tien<4>(params.xDesc->GetLengths());

            unsigned int in_nstride_ = c_ * h_ * w_;
            unsigned int in_cstride_ = h_ * w_;

            kernel(params.x,
                   params.y,
                   params.estimatedMean,
                   params.estimatedVariance,
                   params.bnScale,
                   params.bnBias,
                   params.epsilon,
                   n_,
                   in_cstride_,
                   in_nstride_);
        };
    };

    return result;
}

} // namespace batchnorm

} // namespace solver

} // namespace miopen
