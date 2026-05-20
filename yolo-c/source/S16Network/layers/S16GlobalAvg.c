#include <assert.h>
#include "S16GlobalAvg.h"

CAPI void S16_GlobalAvgForward(void* userdata, const S16Tensor* input, S16Tensor* output)
{
	assert(output->w == 1);
	assert(output->h == 1);
	assert(output->c == input->c);

	const s16* pin = input->ptr;
	s16* pout = output->ptr;
	const int wh = input->w * input->h;
	int i, j;
	for (i = 0; i < output->c; i++) {
		int sum = 0;
		for (j = 0; j < wh; j++) {
			sum += *pin;
			pin++;
		}
		*pout = (s16)(sum / wh);
		pout++;
	}
}