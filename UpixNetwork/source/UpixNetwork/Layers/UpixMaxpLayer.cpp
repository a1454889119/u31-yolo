#include <assert.h>
#include "UpixBaseLayer.h"

template<typename T>
static inline T max2(const T* p)
{
	T vmax = p[0];
	if (p[1] > vmax) vmax = p[1];
	return vmax;
}

template<typename T>
static inline T max3(const T* p)
{
	T vmax = p[0];
	if (p[1] > vmax) vmax = p[1];
	if (p[2] > vmax) vmax = p[2];
	return vmax;
}

template<typename T>
static inline T max4(const T* p)
{
	T vmax = p[0];
	if (p[1] > vmax) vmax = p[1];
	if (p[2] > vmax) vmax = p[2];
	if (p[3] > vmax) vmax = p[3];
	return vmax;
}

template<typename T>
static inline T max5(const T* p)
{
	T vmax = p[0];
	if (p[1] > vmax) vmax = p[1];
	if (p[2] > vmax) vmax = p[2];
	if (p[3] > vmax) vmax = p[3];
	if (p[4] > vmax) vmax = p[4];
	return vmax;
}

template<typename T>
static inline T max6(const T* p)
{
	T vmax = p[0];
	if (p[1] > vmax) vmax = p[1];
	if (p[2] > vmax) vmax = p[2];
	if (p[3] > vmax) vmax = p[3];
	if (p[4] > vmax) vmax = p[4];
	if (p[5] > vmax) vmax = p[5];
	return vmax;
}

template<typename T>
static inline T max7(const T* p)
{
	T vmax = p[0];
	if (p[1] > vmax) vmax = p[1];
	if (p[2] > vmax) vmax = p[2];
	if (p[3] > vmax) vmax = p[3];
	if (p[4] > vmax) vmax = p[4];
	if (p[5] > vmax) vmax = p[5];
	if (p[6] > vmax) vmax = p[6];
	return vmax;
}

template<typename T>
static inline T max8(const T* p)
{
	T vmax = p[0];
	if (p[1] > vmax) vmax = p[1];
	if (p[2] > vmax) vmax = p[2];
	if (p[3] > vmax) vmax = p[3];
	if (p[4] > vmax) vmax = p[4];
	if (p[5] > vmax) vmax = p[5];
	if (p[6] > vmax) vmax = p[6];
	if (p[7] > vmax) vmax = p[7];
	return vmax;
}

template<typename T>
static void maxpool(
	const std::vector<T>& input, 
	std::vector<T>& output, 
	const TensorShape& shape, 
	int ksize, int stride, int pad,
	T minv)
{
	const T* pstart, *pin = input.data();
	T* pout = output.data();
	T vmax;
	int i, j, k, ii;

	for (k = 0; k < shape.c; k++, pin += shape.wh) {
		for (i = 0; i < shape.h; i += stride) {
			int ystart = i - pad;
			int yend = ystart + ksize - 1;
			int ysteps;
			if (ystart < 0) ystart = 0;
			if (yend >= shape.h) yend = shape.h - 1;
			ysteps = yend - ystart + 1;

			pstart = pin + ystart * shape.w;
			for (j = 0; j < shape.w; j += stride, pout++) {
				const T* ppstart;
				int xstart = j - pad;
				int xend = xstart + ksize - 1;
				int xsteps;

				if (xstart < 0) xstart = 0;
				if (xend >= shape.w) xend = shape.w - 1;
				xsteps = xend - xstart + 1;

				ppstart = pstart + xstart;
				vmax = minv;
				switch (xsteps) {
				case 0: break;
				case 2:
					for (ii = 0; ii < ysteps; ii++, ppstart += shape.w) {
						T maxr = max2<T>(ppstart);
						if (maxr > vmax)
							vmax = maxr;
					}
					break;
				case 3:
					for (ii = 0; ii < ysteps; ii++, ppstart += shape.w) {
						T maxr = max3<T>(ppstart);
						if (maxr > vmax)
							vmax = maxr;
					}
					break;

				case 4:
					for (ii = 0; ii < ysteps; ii++, ppstart += shape.w) {
						T maxr = max4<T>(ppstart);
						if (maxr > vmax)
							vmax = maxr;
					}
					break;

				case 5:
					for (ii = 0; ii < ysteps; ii++, ppstart += shape.w) {
						T maxr = max5<T>(ppstart);
						if (maxr > vmax)
							vmax = maxr;
					}
					break;

				case 6:
					for (ii = 0; ii < ysteps; ii++, ppstart += shape.w) {
						T maxr = max6<T>(ppstart);
						if (maxr > vmax)
							vmax = maxr;
					}
					break;

				case 7:
					for (ii = 0; ii < ysteps; ii++, ppstart += shape.w) {
						T maxr = max7<T>(ppstart);
						if (maxr > vmax)
							vmax = maxr;
					}
					break;

				case 8:
					for (ii = 0; ii < ysteps; ii++, ppstart += shape.w) {
						T maxr = max8<T>(ppstart);
						if (maxr > vmax)
							vmax = maxr;
					}
					break;
				default: assert(0); break;
				}

				*pout = vmax;
			}
		}
	}

}


void UpixMaxpLayer::forward_float()
{
	maxpool<float>(_input.fdata, _output.fdata, _input.shape, _info->ksize, _info->stride, _info->pad, -FLT_MAX);
}

void UpixMaxpLayer::forward_int()
{
	maxpool<int>(_input.idata, _output.idata, _input.shape, _info->ksize, _info->stride, _info->pad, SHRT_MIN);
}
