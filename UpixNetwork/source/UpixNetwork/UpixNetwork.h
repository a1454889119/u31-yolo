#ifndef __UPIX_NETWORK_H__
#define __UPIX_NETWORK_H__

#include "UpixNetworkCommon.h"



class UpixBaseLayer;
class UpixNetworkPrivate;
class UpixNetwork
{
public:
	UpixNetwork();
	~UpixNetwork();

	bool loadNetwork(const char* csv_file);
	bool loadWeight(const char* weight_file);

	void quant16();
	bool loadQuant(const char* maxmin_file);

	TensorShape inputShape()const;

	const std::vector<UpixBaseLayer*>& layers()const;

	TensorShape output(int idx)const;

	void forward(const float* input_data);

	void dumpQuantedWeights(const char* filename);

	void forward(const int* input_data);

private:
	UpixNetworkPrivate* d;
};


#endif // !__UPIX_NETWORK_H__
