#ifndef __RECT_DECODER_160X120_H__
#define __RECT_DECODER_160X120_H__

typedef struct stTarget2Type {
	int type;
	int score;
	int x; // left
	int y; // top
	int w;
	int h;
}Target2Type;

typedef struct stRectDecoder RectDecoder;

#ifdef __cplusplus
extern "C" {
#endif

const Target2Type* get_decoded_rects_160x120();

int decode_rects_160x120(
	const short* output49, const short* output54,
	int w0, int h0,
	const short* output59, const short* output64,
	int w1, int h1
);

#ifdef __cplusplus
}
#endif



#endif // !__RECT_DECODER_160X120_H__
