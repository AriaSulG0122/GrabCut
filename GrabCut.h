#pragma once
#include <cv.h>
#include <iostream>

enum
{
	GC_WITH_RECT  = 0, //����ȡ������Ҫ��ʼ��GMM
	GC_CUT        = 1  //����ȡ��������Ҫ�ٳ�ʼ��GMM
};
//����ʾʱ��ȷ�������Ĳ��ֽ��ᱻ���֣����ಿ��������ʾ
enum {
	MUST_BGD = 0,//ȷ������
	MUST_FGD = 1,//ȷ��ǰ��
	MAYBE_BGD = 2,//���ܱ���
	MAYBE_FGD = 3//����ǰ��
};

class GrabCut2D
{
public:
	void GrabCut( cv::InputArray _img, cv::InputOutputArray _mask, cv::Rect rect,
		cv::InputOutputArray _bgdModel,cv::InputOutputArray _fgdModel,int mode);  

	~GrabCut2D(void);
};




