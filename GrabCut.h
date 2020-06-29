#pragma once
#include <cv.h>
#include <iostream>

enum
{
	GC_WITH_RECT  = 0, //初次取样，需要初始化GMM
	GC_CUT        = 1  //后续取样，不需要再初始化GMM
};
//在显示时，确定背景的部分将会被遮罩；其余部分正常显示
enum {
	MUST_BGD = 0,//确定背景
	MUST_FGD = 1,//确定前景
	MAYBE_BGD = 2,//可能背景
	MAYBE_FGD = 3//可能前景
};

class GrabCut2D
{
public:
	void GrabCut( cv::InputArray _img, cv::InputOutputArray _mask, cv::Rect rect,
		cv::InputOutputArray _bgdModel,cv::InputOutputArray _fgdModel,int mode);  

	~GrabCut2D(void);
};




