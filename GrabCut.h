#pragma once
#include <cv.h>
#include <iostream>
#include <limits>
#include <vector>
#include "graph.h"
#include "GMM.h"

using namespace cv;
using namespace std;
typedef Graph<int,int,int> GraphType;

enum
{
	GC_WITH_RECT  = 0, //初次取样，需要初始化GMM
	GC_CUT        = 1  //后续取样，不需要再初始化GMM
};
//在显示时，确定背景与可能背景的部分将会被遮罩；其余部分正常显示
enum {
	MUST_BGD = 0,//确定背景
	MUST_FGD = 1,//确定前景
	MAYBE_BGD = 2,//可能背景
	MAYBE_FGD = 3//可能前景
};

class MyGrabCut
{
public:
	void GrabCut( cv::InputArray _img, cv::InputOutputArray _mask, cv::Rect rect,
		cv::InputOutputArray _bgdModel,cv::InputOutputArray _fgdModel,int mode);  
	~MyGrabCut(void);
private:
	//计算 Beta 的值(根据论文中的公式5)
	double calBeta(const Mat& _img);
	//计算平滑项
	void calcuNWeight(const Mat& _img, Mat& _l, Mat& _ul, Mat& _u, Mat& _ur, double _beta, double _gamma);
	//利用opencv中的 kmeans 方法初始化 GMM 模型
	void initGMMs(const Mat& img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM);
	//迭代循环第一步，为每个像素分配GMM中所属的高斯模型，保存在partIndex中。
	void assignGMM(const Mat& _img, const Mat& _mask, const GMM& _bgdGMM, const GMM& _fgdGMM, Mat& _GmmNumber);
	//迭代循环第二步，根据得到的结果计算GMM参数值。
	void learnGMMs(const Mat& _img, const Mat& _mask, GMM& _bgdGMM, GMM& _fgdGMM, const Mat& _GmmNumber);
	//根据得到的结果构造图（调用给的轮子）
	void getGraph(const Mat& _img, const Mat& _mask, const GMM& _bgdGMM, const GMM& _fgdGMM, const Mat& _l, const Mat& _ul, const Mat& _u, const Mat& _ur, GraphType *_graph);
	//进行分割
	void estimateSegmentation(GraphType *_graph, Mat& _mask);
};




