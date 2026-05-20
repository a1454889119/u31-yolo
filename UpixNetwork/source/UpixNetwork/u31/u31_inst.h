#ifndef __U31_INSTRUCTIONS_H__
#define __U31_INSTRUCTIONS_H__

#ifndef CAPI
#ifdef __cplusplus
#define CAPI extern "C"
#else
#define CAPI
#endif
#endif

CAPI int u31_exp2_m16p16(int x);
CAPI int u31_exp_m22p10(int x);
CAPI int u31_sigmoid(int x);


#endif // !__U31_INSTRUCTIONS_H__
