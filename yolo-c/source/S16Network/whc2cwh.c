
void whc2cwh(const short* input, short* output, int w, int h, int c)
{
	const short* inptr = input;
	int wh = w * h;
	int i, j, k;
	for (i = 0; i < h; i++) {
		for (j = 0; j < w; j++) {
			const short* cptr = inptr;
			for (k = 0; k < c; k++, output++, cptr += wh) {
				*output = *cptr;
			}
			inptr++;
		}
	}
}
