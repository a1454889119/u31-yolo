#ifndef __S16_CNN_COMMON_H__
#define __S16_CNN_COMMON_H__

#ifdef _MSC_VER
#pragma warning(disable:4996)
#endif

#ifndef __cplusplus
#ifdef _MSC_VER
#define inline __inline
#endif
#endif

#ifndef NULL
#define NULL ((void*)0)
#endif

#ifndef CAPI
#ifdef __cplusplus
#define CAPI extern "C"
#else
#define CAPI
#endif
#endif

#define S16_QUANT_SHIFT (10)
#define S16_QUANT_MULT (1 << S16_QUANT_SHIFT)

#ifndef S16
#define S16
typedef signed short s16;
#endif // !S16

static inline s16 s16_trim(int i)
{
//	if (i >= 32767)
//		i = 32767;
//	else if (i <= -32768)
//		i = -32768;

	return (s16)i;
}

static inline s16 s16_add(s16 s1, s16 s2) 
{
	int i = s1 + s2;
	return s16_trim(i);
}

static inline s16 s16_sub(s16 s1, s16 s2)
{
	int i = s1 - s2;
	return s16_trim(i);
}

static inline s16 s16_mad(s16 s1, s16 s2, s16 s3)
{
	int i = s1 + s2 * s3;
	return s16_trim(i);
}

static inline s16 s16_madm(s16 s1, s16 s2, s16 s3, s16 s4)
{
	int i = s1 * s2 + s3 * s4;
	return s16_trim(i);
}

static inline float s16_2_float(s16 s) { return (float)s / S16_QUANT_MULT; }
static inline s16 float_2_s16(float f) { return (s16)(f * S16_QUANT_MULT + 0.5f); }



typedef struct stQuantize {
	int scale;		// 用于de-quantize时乘
	int zero_point;
	int c_start;
	int c_end;
}Quantize;
#define MAX_QUANTIZE_COUNT 8

typedef struct stS16Tensor {
	int w;
	int h;
	int c;

	int whc;

	s16 *ptr;
	Quantize quants[MAX_QUANTIZE_COUNT];
	int quant_count;
}S16Tensor;

#endif