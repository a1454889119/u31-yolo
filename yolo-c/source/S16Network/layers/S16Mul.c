#include <assert.h>
#include "S16Mul.h"

CAPI void S16_MulForward(void* userdata, const S16Tensor* input, S16Tensor* output)
{
	assert(input->w == output->w);
	assert(input->h == output->h);
	assert(input->c == output->c * 2);
	int i, j;
	const s16* pin = input->ptr;
	const s16* pin_k = pin + output->whc;
	s16* pout = output->ptr;
	for (i = 0; i < output->c; i++) {
		for (j = 0; j < output->w*output->h; j++) {
			int v = *pin * *pin_k;
			*pout = (s16)(v >> S16_QUANT_SHIFT);

			pin++;
			pout++;
		}
		pin_k++;
	}
}