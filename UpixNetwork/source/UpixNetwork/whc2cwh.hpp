#ifndef __WHC2CWH_HPP__
#define __WHC2CWH_HPP__

template<typename T>
void whc2cwh(const T* input, T* output, int w, int h, int c)
{
	const T* inptr = input;
	int wh = w * h;
	int i, j, k;
	for (i = 0; i < h; i++) {
		for (j = 0; j < w; j++) {
			const T* cptr = inptr;
			for (k = 0; k < c; k++, output++, cptr += wh) {
				*output = *cptr;
			}
			inptr++;
		}
	}
}

#endif // !__WHC2CWH_HPP__
