#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "GrabCut.h"
#include <iostream>
using namespace std;
using namespace cv;

//不同颜色，用于用户的交互输出
const Scalar BLUE = Scalar(255,0,0); // Background 
const Scalar GREEN = Scalar(0,255,0);//Foreground
const Scalar LIGHTBLUE = Scalar(255,255,160);//ProbBackground
const Scalar PINK = Scalar(230,130,255); //ProbBackground
const Scalar RED = Scalar(0,0,255);//color of Rectangle

const int BGD_KEY = CV_EVENT_FLAG_CTRLKEY;// When press "CTRL" key,the value of flags return.
const int FGD_KEY = CV_EVENT_FLAG_SHIFTKEY;// When press "SHIFT" key, the value of flags return.




class GCApplication
{
public:
	enum{ NOT_SET = 0, IN_PROCESS = 1, SET = 2 };//与用户交互框的三种状态
	static const int radius = 2;
	static const int thickness = -1;

	void reset();
	void setImageAndWinName( const Mat& _image, const string& _winName );
	void showImage();
	void mouseClick( int event, int x, int y, int flags, void* param );
	int nextIter();
	int getIterCount() const { return iterCount; }
	void getBinMask(const Mat& comMask, Mat& binMask);
private:
	void setRectInMask();
	void setLblsInMask( int flags, Point p, bool isPr );

	const string* winName;
	const Mat* image;
	Mat mask, alphaMask;
	Mat bgdModel, fgdModel;

	uchar rectState, lblsState, prLblsState;
	bool isInitialized;

	Rect rect;
	vector<Point> fgdPxls, bgdPxls, prFgdPxls, prBgdPxls;
	int iterCount;
	GrabCut2D gc;
};


