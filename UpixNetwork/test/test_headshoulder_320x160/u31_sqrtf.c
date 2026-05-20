#include <math.h>
#include "u31_sqrtf.h"

#define CHECK_BIT(var, pos) ((var) & (1 << (pos)))   // 获取指定比特位对应的值(10进制)
#define f_infinity_n       0xFF800000  //1 11111111 00000000000000000000000 -inf 负无穷大    
#define f_infinity         0x7F800000  //0 11111111 00000000000000000000000 +inf 正无穷大
#define f_0                0x00000000  //0 00000000 00000000000000000000000 0.000000000  正0    
#define f_0_n              0x80000000  //1 00000000 00000000000000000000000 -0.000000000 负0

static int is_inf(unsigned int hex)
{
	return ((f_infinity_n == hex) || (f_infinity == hex));
}

typedef union
{
    unsigned int word;
    int sword;
    unsigned short hword[2];
    short shword[2];
    unsigned char byte[4];
    float f;
} Reg32;

static void u31_op_sqrtf(Reg32* dst, Reg32* opt1)
{
    int nHat = 0;
    int b = 32768;
    int bshft = 15;
    int valid;

    for (int i = 0; i < 16; ++i)
    {
        int tmps = ((nHat * 2) + b);
        int tmp = tmps << bshft;

        valid = (opt1->sword >= tmp) ? 1 : 0;

        if (valid)
        {
            nHat += b;
            opt1->sword -= tmp;
        }

        b >>= 1;

        bshft--;
    }
    dst->sword = nHat * 46341;
}

static void u31_op_fsqrt(Reg32* dst, Reg32* opt1)
{
    if (CHECK_BIT(opt1->word, 31))
    {
        dst->word = f_0;
        return;
    }
    if (is_inf(opt1->word))
    {
        dst->word = opt1->word;
        return;
    }
    int n;
    float m, exponent, mantissa;
    m = frexpf(opt1->f, &n);

    if (n % 2)
    {
        m /= 2;
        n += 1;
    }
    // 先计算指数开方
    exponent = powf(2.0f, n * 0.5f);
    // 计算尾数开方 先转成整数
    m *= powf(2, 31);
    Reg32 tmp, value;
    value.f = m;

    u31_op_sqrtf(&tmp, &value);
    mantissa = tmp.word / powf(2.0f, 31.0f);
    dst->f = exponent * mantissa;
}

CAPI float u31_sqrtf(float x)
{
    Reg32 input, output;
    input.f = x;
    u31_op_fsqrt(&output, &input);
	return output.f;
}
