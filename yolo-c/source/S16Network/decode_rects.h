#ifndef __RECT_DECODER_H__
#define __RECT_DECODER_H__

typedef struct stTargetRect {
	int type;
	int score;
	int x; // left
	int y; // top
	int w;
	int h;
}TargetRect;

typedef struct stRectDecoder RectDecoder;

#ifdef __cplusplus
extern "C" {
#endif

const TargetRect* get_decoded_rects();

int decode_rects(
	const short* c3_16_0, const short* c3_16_1, const short* c12_16,
	const short* c3_32_0, const short* c3_32_1, const short* c12_32
);

void whc2cwh(const short* input, short* output, int w, int h, int c);

#ifdef __cplusplus
}
#endif



#endif // !__RECT_DECODER_H__
