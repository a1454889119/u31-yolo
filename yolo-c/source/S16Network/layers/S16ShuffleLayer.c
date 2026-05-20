#include <assert.h>
#include <memory.h>
#include "S16ShuffleLayer.h"

CAPI void S16_ShuffleForward(void* userdata, const S16Tensor* input, S16Tensor* output)
{
	int wh = input->w * input->h;
	const s16* in_ptr = input->ptr;
	s16* out_ptr0 = output->ptr;
	s16* out_ptr1 = out_ptr0 + wh * (output->c / 2);

	assert(input->w == output->w);
	assert(input->h == output->h);
	assert(input->c == output->c);
	assert(input->c % 2 == 0);

	int i;
	for (i = 0; i < input->c; i += 2) {
		memcpy(out_ptr0, in_ptr, wh * sizeof(s16));
		in_ptr += wh;
		out_ptr0 += wh;

		memcpy(out_ptr1, in_ptr, wh * sizeof(s16));
		in_ptr += wh;		
		out_ptr1 += wh;
	}
}