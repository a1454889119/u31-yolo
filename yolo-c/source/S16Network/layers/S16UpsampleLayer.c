#include <assert.h>
#include <memory.h>
#include "S16UpsampleLayer.h"

CAPI void S16_UpsampleForward(void* userdata, const S16Tensor* input, S16Tensor* output)
{
	S16UpsampleLayer* up = (S16UpsampleLayer*)userdata;
	const s16* in_ptr = input->ptr;
	s16* out_ptr = output->ptr;
	assert(input->w * up->ksize == output->w);
	assert(input->h * up->ksize == output->h);
	assert(input->c == output->c);

	int i, j, k;
	for (i = 0; i < input->c; i++) {
		for (j = 0; j < input->h; j++) {
			int t;
			s16* out_first = out_ptr;

			for (k = 0; k < input->w; k++) {
				for (t = 0; t < up->ksize; t++) {
					*out_ptr++ = *in_ptr;
				}
				in_ptr++;
			}
			
			for (t = 1; t < up->ksize; t++) {
				memcpy(out_ptr, out_first, sizeof(s16)*output->w);
				out_ptr += output->w;
			}
		}
	}
}