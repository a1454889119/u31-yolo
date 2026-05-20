#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include "S16Add.h"

static void activate_linear(const int* pin, s16* pout, int count)
{
	const int* pin_end = pin + count;
	while (pin < pin_end) {
		*pout = (s16 )*pin;
		pin++;
		pout++;
	}
}

static void activate_leaky(const int* pin, s16* pout, int count)
{
	const int* pin_end = pin + count;
	while (pin < pin_end) {
		int res = *pin;
		if (res < 0) {
			res >>= 10;
		}
		*pout = (s16)res;
		pin++;
		pout++;
	}
}

static void activate_relu(const int* pin, s16* pout, int count)
{
	const int* pin_end = pin + count;
	while (pin < pin_end) {
		int res = *pin;
		if (res < 0) {
			res = 0;
		}
		*pout = (s16)res;
		pin++;
		pout++;
	}
}

CAPI void S16_AddForward(void* userdata, const S16Tensor* input, S16Tensor* output)
{
	S16AddLayer* add = (S16AddLayer*)userdata;
	int count = input->whc;
	int parts = input->c / output->c;
	int* sum_data = (int*)malloc(sizeof(int) * output->whc);
	assert(input->w == output->w);
	assert(input->h == output->h);
	assert(parts > 1 && parts * output->c == input->c);
	assert(sum_data);

	memset(sum_data, 0, sizeof(int) * output->whc);


	const s16* pin = input->ptr;
	for (int i = 0; i < parts; i++) {
		int* psum = sum_data;
		for (int j = 0; j < output->whc; j++, pin) {
			psum[j] += (int)pin[j];
		}
		pin += output->whc;
	}

	switch (add->activate_type) {
	case ActivateType_Linear: activate_linear(sum_data, output->ptr, output->whc); break;
	case ActivateType_Leaky: activate_leaky(sum_data, output->ptr, output->whc); break;
	case ActivateType_Relu: activate_relu(sum_data, output->ptr, output->whc); break;
	default: assert(0);
	}


	free(sum_data);
}
