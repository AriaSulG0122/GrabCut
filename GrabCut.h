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
	GC_WITH_RECT  = 0, //����ȡ������Ҫ��ʼ��GMM
	GC_CUT        = 1  //����ȡ��������Ҫ�ٳ�ʼ��GMM
};
//����ʾʱ��ȷ����������ܱ����Ĳ��ֽ��ᱻ���֣����ಿ��������ʾ
enum {
	MUST_BGD = 0,//ȷ������
	MUST_FGD = 1,//ȷ��ǰ��
	MAYBE_BGD = 2,//���ܱ���
	MAYBE_FGD = 3//����ǰ��
};

class MyGrabCut
{
public:
	void GrabCut( cv::InputArray _img, cv::InputOutputArray _mask, cv::Rect rect,
		cv::InputOutputArray _bgdModel,cv::InputOutputArray _fgdModel,int mode);  
	~MyGrabCut(void);
private:
	//���� Beta ��ֵ(���������еĹ�ʽ5)
	double calBeta(const Mat& _img);
	//����ƽ����
	void calcuNWeight(const Mat& _img, Mat& _l, Mat& _ul, Mat& _u, Mat& _ur, double _beta, double _gamma);
	//����opencv�е� kmeans ������ʼ�� GMM ģ��
	void initGMMs(const Mat& img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM);
	//����ѭ����һ����Ϊÿ�����ط���GMM�������ĸ�˹ģ�ͣ�������partIndex�С�
	void assignGMM(const Mat& _img, const Mat& _mask, const GMM& _bgdGMM, const GMM& _fgdGMM, Mat& _GmmNumber);
	//����ѭ���ڶ��������ݵõ��Ľ������GMM����ֵ��
	void learnGMMs(const Mat& _img, const Mat& _mask, GMM& _bgdGMM, GMM& _fgdGMM, const Mat& _GmmNumber);
	//���ݵõ��Ľ������ͼ�����ø������ӣ�
	void getGraph(const Mat& _img, const Mat& _mask, const GMM& _bgdGMM, const GMM& _fgdGMM, const Mat& _l, const Mat& _ul, const Mat& _u, const Mat& _ur, GraphType *_graph);
	//���зָ�
	void estimateSegmentation(GraphType *_graph, Mat& _mask);
};




