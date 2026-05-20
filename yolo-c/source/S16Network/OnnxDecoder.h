#ifndef __ONNX_RECT_DECODER_H__
#define __ONNX_RECT_DECODER_H__

#include <vector>

typedef struct stOnnxTarget {
	int type;
	int score;
	int x; // left
	int y; // top
	int w;
	int h;
}OnnxTarget;

class OnnxDecoderPrivate;
class OnnxDecoder {
public:
	OnnxDecoder(float score_thres);
	~OnnxDecoder();

	void reset();

	int decode(const short* c2, const short* c40, int w, int h, int scale);

	const std::vector<OnnxTarget>& getTargets()const;

private:
	OnnxDecoderPrivate* d;
};


#endif // !__ONNX_RECT_DECODER_H__
