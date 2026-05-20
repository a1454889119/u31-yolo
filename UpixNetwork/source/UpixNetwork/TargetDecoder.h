#ifndef __TARGET_DECODER_H__
#define __TARGET_DECODER_H__

#include <vector>

typedef struct stTarget {
	int type;
	float score;
	float x; // left
	float y; // top
	float w;
	float h;
}Target;

class TargetDecoderPrivate;
class TargetDecoder {
public:
	TargetDecoder();
	~TargetDecoder();

	void reset(float score_thres);

	int decode(  const int* scores,	  const int* c40, int w, int h, int typecount, int scale);
	int decode(const float* scores, const float* c40, int w, int h, int typecount, int scale);

	const std::vector<Target>& getTargets()const;

private:
	TargetDecoderPrivate* d;
};

#endif // !__TARGET_DECODER_H__
