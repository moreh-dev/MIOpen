/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

# include <miopen/maskedfill/solvers.hpp>

# include <miopen/maskedfill/invoke_params.hpp>

# include <miopen/kernel_build_params.hpp>

# define LOCAL_SIZE 256

namespace miopen :: solver :: maskedfill {

	bool MaskedFill :: IsApplicable(ExecutionContext const & context, miopen :: maskedfill :: ProblemDescription const & problem) const {
		return true;
	}

	ConvSolution MaskedFill :: GetSolution(ExecutionContext const & context, miopen :: maskedfill :: ProblemDescription const & problem) const {
		auto result = ConvSolution {miopenStatusSuccess};
		{
			auto const dtype = problem.GetInputDesc().GetType();
			auto const inputlengths = problem.GetInputDesc().GetLengths();
			auto const numel = std :: accumulate(inputlengths.begin(), inputlengths.end(), 1, std :: multiplies<> {});
			size_t xlocalsize = LOCAL_SIZE;
			size_t ylocalsize = 1;
			size_t zlocalsize = 1;
			size_t xgridsize  = AlignUp(numel, xlocalsize);
			size_t ygridsize  = 1;
			size_t zgridsize  = 1;
			auto kernel = KernelInfo {};
			kernel.kernel_file = "MIOpenMaskedFill.CPP";
			kernel.kernel_name = problem.IsBackward()? "MaskedFillBackward" : "MaskedFillForward";
			auto const buildparams = KernelBuildParameters {
				{"MIOPEN_USE_FP16",		static_cast<int>(dtype == miopenHalf)},
				{"MIOPEN_USE_FP32",		static_cast<int>(dtype == miopenFloat)},
				{"MIOPEN_USE_FP64",		static_cast<int>(dtype == miopenDouble)},
				{"MIOPEN_USE_BFP16",	static_cast<int>(dtype == miopenBFloat16)},
				{"LOCAL_SIZE",			LOCAL_SIZE},
			};
			kernel.comp_options = buildparams.GenerateFor(kbp :: HIP {});
			kernel.l_wk.push_back(xlocalsize);
			kernel.l_wk.push_back(ylocalsize);
			kernel.l_wk.push_back(zlocalsize);
			kernel.g_wk.push_back(xgridsize);
			kernel.g_wk.push_back(ygridsize);
			kernel.g_wk.push_back(zgridsize);
			result.construction_params.push_back(kernel);
		}
		result.invoker_factory = [] (std :: vector<Kernel> const & kernels) {
			return [=] (Handle const & handle, AnyInvokeParams const & rawparams) {
				decltype(auto) kernel = handle.Run(kernels.front());
				decltype(auto) params = rawparams.CastTo<miopen :: maskedfill :: InvokeParams>();
				auto const inputlengths = params.inputDesc -> GetLengths();
				auto const numel = std :: accumulate(inputlengths.begin(), inputlengths.end(), 1, std :: multiplies<> {});
				kernel(
					params.input,
					params.output,
					params.mask,
					params.value,
					numel
				);
			};
		};
		return result;
	}

}
