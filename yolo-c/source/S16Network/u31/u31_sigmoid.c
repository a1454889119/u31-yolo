#include <math.h>
#include "u31_inst.h"

static double __exp2(int x)
{
	static const int exp_lut[32] = {
		16562, 16925, 17295, 17674, 18061, 18456, 18861, 19274,
		19696, 20127, 20568, 21018, 21478, 21949, 22429, 22920,
		23422, 23935, 24459, 24995, 25542, 26102, 26673, 27257,
		27854, 28464, 29087, 29724, 30375, 31040, 31720, 32415,
	};
	double ret = 0.0;
	char integer = x >> 24;             // 取m8p24格式的整数部分
	int remain_bin = x & 0xffffff;    // 取m8p24格式的小数部分二进制对应的10进制数
	int index = (x & 0xf80000) >> 19; // 取m8p24格式的小数部分前5个二进制数对应的10进制数
	__int64 dat_integer, dat_remain, mul_dat;

	if (0x000000 != remain_bin) // 判断是不是整数
	{
		if (x < 0) // 判断符号
		{
			dat_integer = (__int64)(floor(pow(2, integer) * pow(2, 28))); // p28
			dat_remain = exp_lut[index];                                                  // p14
			mul_dat = (__int64)(floor((double)(dat_integer)*dat_remain / pow(2, 26)));
		}
		else
		{
			dat_integer = (__int64)(floor(pow(2, integer) * pow(2, 16))); // p16
			dat_remain = exp_lut[index];                                                  // p14
			mul_dat = (__int64)(floor((double)(dat_integer)*dat_remain / pow(2, 14)));
		}
		ret = (double)mul_dat;
	}
	else
	{
		ret = pow(2, integer) * pow(2, 16);
	}
	ret = ret / pow(2, 16);
	// printf("\tinteger=%d remain_bin=0x%08x index=%d dat_integer=%ld dat_remain=%ld mul_dat=%d ret=%lf\n", integer, remain_bin, index, dat_integer, dat_remain, mul_dat, ret);

	return ret;
}

CAPI int u31_exp2_m16p16(int x)
{

	/**
	 * m16p16数据格式就要求输入的浮点数在-16~15之间，否则2的指数后转成m16p16就会溢出(2^15*2^16=2^31),
	 * 这样就要求输入的m8p24必须在-16*2^24~15*2^24之间
	 */
	if (x <= -16 * pow(2, 24))
	{
		return 1;
	}
	if (15 * pow(2, 24) <= x)
	{
		return 0x7fffffff;
	}
	double ret = __exp2(x);
	ret *= pow(2, 16);

	if (ret > 0x7fffffff)
		return 0x7fffffff;

	return (int)ret;
}

static const int exp_k = 1477;
static const int max_sigmoid = 7168;
static const int min_sigmoid = -7168;
static const int m22p10_1 = 1024;
static const int m22p10_1024 = 1024 * 1024;


CAPI int u31_exp_m22p10(int v)
{
	v *= exp_k;
	v <<= 4;
	v = u31_exp2_m16p16(v);
	v >>= 6;
	return v;
}

CAPI int u31_sigmoid(int v)
{
	v = -v;
	if (v >= max_sigmoid) {
		return 0;
	}
	if (v <= min_sigmoid) {
		return m22p10_1;
	}
	v = u31_exp_m22p10(v);
	v += m22p10_1;
	v = m22p10_1024 / v;
	return v;
}