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

# include <miopen/miopen.h>
# include <gtest/gtest.h>
# include <miopen/maskedfill.hpp>
# include "cpu_maskedfill.hpp"

# include "get_handle.hpp"
# include "tensor_holder.hpp"
# include "random.hpp"
# include "verify.hpp"
# include "../driver/tensor_driver.hpp"

struct MaskedFillTestCase /* MaskedFillTestParameters */ {
	std :: vector<size_t> const size; // or "dims"
	std :: vector<size_t> const & GetSize() const { return size; };
	friend std :: ostream const & operator<< (std :: ostream & os, MaskedFillTestCase const & parameters) {
		os << "{{";
		for (auto dimension = 0; dimension < parameters.size.size(); ++dimension) {
			os << parameters.size[dimension];
			if (dimension < parameters.size.size() - 1)
				os << ", ";

		}
		return os << "}}";
	}
};
std :: vector<MaskedFillTestCase> const MaskedFillTestConfigs(miopenMaskedFillDirection_t const direction) {
	switch (direction) {
		case MIOPEN_MASKEDFILL_FORWARD:
		return {
			{{1}},
			{{2, 2}},
			{{1323, 12, 12, 20}},
		};
		break;
		case MIOPEN_MASKEDFILL_BACKWARD:
		return {
			{{1}},
			{{2, 2}},
			{{1323, 12, 12, 20}},
		};
	}
}

inline int SetTensorLayout(miopen :: TensorDescriptor & desc) {
	auto const lengths = desc.GetLengths();
	std :: vector<int> intlengths {lengths.begin(), lengths.end()};
	return SetTensorNd(& desc, intlengths, desc.GetType());
}

template <typename T = float> class MaskedFillTest: public testing :: TestWithParam<MaskedFillTestCase> {
	miopenMaskedFillDirection_t const direction;
	tensor<T>								input,		output,		ref_output;
	tensor<int8_t>																	mask;	// `tensor<bool>`s aren't implemented (because `miopen_type<bool>` isn't implemented)
	miopen :: Allocator :: ManageDataPtr	input_dev,	output_dev,				mask_dev;
	T value;

	void SetUp() override {
		auto && handle = get_handle();
		auto size = GetParam().GetSize();

		auto gen_value = [] (auto ...) { return prng :: gen_descreet_uniform_sign<T>(1e-2, 100); };
		input = tensor<T> {size}.generate(gen_value);
		SetTensorLayout(input.desc);
		input_dev = handle.Write(input.data);

		output = tensor<T> {size};
		SetTensorLayout(output.desc);
		std :: fill(output.begin(), output.end(), std :: numeric_limits<T> :: quiet_NaN());
		output_dev = handle.Write(output.data);

		ref_output = tensor<T> {size};
		std :: fill(ref_output.begin(), ref_output.end(), std :: numeric_limits<T> :: quiet_NaN());

		std :: mt19937 generator;
		std :: uniform_int_distribution<unsigned int> distribution {0, 1};
		mask = tensor<int8_t> {size}.generate([&] (auto ...) { return distribution(generator); });
		mask_dev = handle.Write(mask.data);

		value = gen_value();
	}

	public:
	MaskedFillTest(miopenMaskedFillDirection_t _direction): direction(_direction) {}

	void RunTest() {
		auto && handle = get_handle();

		miopenStatus_t status;
		switch (direction) {
			case MIOPEN_MASKEDFILL_FORWARD:
			status = miopen :: MaskedFillForward(
				handle,

				input.desc,
				input_dev.get(),
				output.desc,
				output_dev.get(),

				mask.desc,
				mask_dev.get(),

				value
			);
			EXPECT_EQ(status, miopenStatusSuccess);
			output.data = handle.Read<T>(output_dev, output.data.size());
			cpu_maskedfill_forward<T>(input, ref_output, mask, value);
			break;
			case MIOPEN_MASKEDFILL_BACKWARD:
			status = miopen :: MaskedFillBackward(
				handle,

				input.desc,
				input_dev.get(),
				output.desc,
				output_dev.get(),

				mask.desc,
				mask_dev.get(),

				value
			);
			EXPECT_EQ(status, miopenStatusSuccess);
			output.data = handle.Read<T>(output_dev, output.data.size());
			cpu_maskedfill_backward<T>(input, ref_output, mask);
		}
	}
	void Verify() const {
		auto error = miopen :: rms_range(output, ref_output);
		EXPECT_TRUE(miopen :: range_distance(output) == miopen :: range_distance(ref_output));
		EXPECT_TRUE(error == 0) << "Outputs do not match each other: `error` = " << error;
	}
};

template <typename T = float> struct MaskedFillForwardTest:		MaskedFillTest<T> {
	MaskedFillForwardTest(): MaskedFillTest<T> {MIOPEN_MASKEDFILL_FORWARD} {}
};

template <typename T = float> struct MaskedFillBackwardTest:	MaskedFillTest<T> {
	MaskedFillBackwardTest(): MaskedFillTest<T> {MIOPEN_MASKEDFILL_BACKWARD} {}
};
