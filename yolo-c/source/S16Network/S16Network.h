#ifndef __S16_NETWORK_H__
#define __S16_NETWORK_H__

typedef struct stS16BaseLayer S16BaseLayer;

#include <map>
#include <vector>

class S16NetworkPrivate;
class S16Network {
public:
	S16Network();
	~S16Network();

	int inputW()const;
	int inputH()const;
	int inputC()const;

	bool loadNetwork(const char* csv_file, int version);
	bool loadWeight(const char* weight_file);

	void forward(const short* input_data);
	void forward_quant8bit(const short* input_data);

	void dumpToDir(const char* dir_name, int flag_8bit)const;

	const short* getLayerOutput(int idx)const;

	const std::vector<S16BaseLayer*>& layers()const;
	const std::map<int, S16BaseLayer*>& layerMap()const;

	void dumpLayer(int idx, const char* filename)const;

	void dumpLayer(int idx0, int idx1, const char* filename)const;

private:
	S16NetworkPrivate* d;
};

#endif // !__S16_NETWORK_H__
