#ifndef __UPIX_LAYER_INFO_H__
#define __UPIX_LAYER_INFO_H__

#include <UpixNetwork/UpixNetworkCommon.h>

enum class QuantIntFlag
{
	None,
	MulSub,
	ScaleOnly
};

struct stQuantInt {
	int mulShift = 0;

	QuantIntFlag quantFlag = QuantIntFlag::None;
	float outScale = 0.0f;
	float outZeroPoint = 0.0f;
};

enum class OutputType
{
	None = 0,
	WHC = 1,
	CWH = 2,
	WHC_CWH = 3
};

class UpixLayerInfo
{
public:
	int idx = -1;
	int inw = -1;
	int inh = -1;
	int inc = -1;
	int outw = -1;
	int outh = -1;
	int outc = -1;
	UpixLayerType type = UpixLayerType::MAX;
	ActivateType activate = ActivateType::Linear;
	int ksize = -1;
	int stride = -1;
	int pad = -1;
	int groups = 1;
	std::vector<stInputSource> isrcs;
	int refcount = 0;

	stQuantInt quantInt;
	OutputType outType = OutputType::None;

public:
	static int load(const char* filename, std::vector<UpixLayerInfo>& layers);
};


#endif // !__UPIX_LAYER_INFO_H__
