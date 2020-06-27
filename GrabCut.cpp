#include "GMM.h"
#include "GrabCut.h"
#include "CutGraph.h"
#include <iostream>
#include <limits>
#include <vector>
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
using namespace cv;
using namespace std;

//���� Beta ��ֵ�����������еĹ�ʽ��5���� 
static double calcuBeta(const Mat& _img) {
	double beta;
	double totalDiff = 0;
	for (int y = 0; y < _img.rows; y++) {
		for (int x = 0; x < _img.cols; x++) {
			Vec3d color = (Vec3d)_img.at<Vec3b>(y, x);
			if (x > 0) {
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y, x - 1);
				totalDiff += diff.dot(diff);
			}
			if (y > 0 && x > 0) {
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y - 1, x - 1);
				totalDiff += diff.dot(diff);
			}
			if (y > 0) {
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y - 1, x);
				totalDiff += diff.dot(diff);
			}
			if (y > 0 && x < _img.cols - 1) {
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y - 1, x + 1);
				totalDiff += diff.dot(diff);
			}
		}
	}
	totalDiff *= 2;
	if (totalDiff <= std::numeric_limits<double>::epsilon()) beta = 0;
	else beta = 1.0 / (2 * totalDiff / (8 * _img.cols*_img.rows - 6 * _img.cols - 6 * _img.rows + 4));
	return beta;
}
//�����������ص�Ȩ�ز���ڶԳ��ԣ��˸�������ֻ��Ҫ�����ĸ����� 
static void calcuNWeight(const Mat& _img, Mat& _l, Mat& _ul, Mat& _u, Mat& _ur, double _beta, double _gamma) {
	const double gammaDiv = _gamma / std::sqrt(2.0f);
	_l.create(_img.size(), CV_64FC1);
	_ul.create(_img.size(), CV_64FC1);
	_u.create(_img.size(), CV_64FC1);
	_ur.create(_img.size(), CV_64FC1);
	for (int y = 0; y < _img.rows; y++) {
		for (int x = 0; x < _img.cols; x++) {
			Vec3d color = (Vec3d)_img.at<Vec3b>(y, x);
			if (x - 1 >= 0) {
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y, x - 1);
				_l.at<double>(y, x) = _gamma*exp(-_beta*diff.dot(diff));
			}
			else _l.at<double>(y, x) = 0;
			if (x - 1 >= 0 && y - 1 >= 0) {
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y - 1, x - 1);
				_ul.at<double>(y, x) = gammaDiv*exp(-_beta*diff.dot(diff));
			}
			else _ul.at<double>(y, x) = 0;
			if (y - 1 >= 0) {
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y - 1, x);
				_u.at<double>(y, x) = _gamma*exp(-_beta*diff.dot(diff));
			}
			else _u.at<double>(y, x) = 0;
			if (x + 1 < _img.cols && y - 1 >= 0) {
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y - 1, x + 1);
				_ur.at<double>(y, x) = gammaDiv*exp(-_beta*diff.dot(diff));
			}
			else _ur.at<double>(y, x) = 0;
		}
	}
}

//����opencv�е� kmeans ������ʼ�� GMM ģ��
static void initGMMs(const Mat& img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM) {
	const int kmeansItCount = 10;//ǰ�������ܹ�10��GMM����
	Mat bgdLabels, fgdLabels;
	vector<Vec3f> bgdSamples, fgdSamples;//Vec3fΪ��ͨ��float���������¼��RGB��ͨ����Ϣ
	Point p;
	for (p.y = 0; p.y < img.rows; p.y++) {
		for (p.x = 0; p.x < img.cols; p.x++) {
			if (mask.at<uchar>(p) == MUST_BGD || mask.at<uchar>(p) == MAYBE_BGD)
				bgdSamples.push_back((Vec3f)img.at<Vec3b>(p));//���뱳������
			else
				fgdSamples.push_back((Vec3f)img.at<Vec3b>(p));//����ǰ������
		}
	}
	//��ǰ����������kmeas���࣬��Ϊ5��
	Mat _bgdSamples((int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0]);
	//��������Ϊ���������ݣ��������������ǩ����ֹ��������ֹģʽ�����������������ȣ����ظ���������ʼ�����ĵ��㷨
	kmeans(_bgdSamples, GMM::K, bgdLabels,
		TermCriteria(CV_TERMCRIT_ITER, kmeansItCount, 0.0), 0, KMEANS_PP_CENTERS);
	//�Ա�����������kmeas���࣬��Ϊ5��
	Mat _fgdSamples((int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0]);
	kmeans(_fgdSamples, GMM::K, fgdLabels,
		TermCriteria(CV_TERMCRIT_ITER, kmeansItCount, 0.0), 0, KMEANS_PP_CENTERS);

	//��ʼ���м����
	bgdGMM.InitInterVar();
	//���ݾ������������ÿ��GMM����еľ�ֵ��Э����Ȳ���
	for (int i = 0; i < (int)bgdSamples.size(); i++)
		//����Ӧģ�����ӵ����㣬addSample��һ������Ϊ��Ӧģ�ͱ�ţ��ڶ�������Ϊ���ص���ɫ��Ϣ
		bgdGMM.addSample(bgdLabels.at<int>(i, 0), bgdSamples[i]);
	bgdGMM.UpdatePara();

	fgdGMM.InitInterVar();
	for (int i = 0; i < (int)fgdSamples.size(); i++)
		fgdGMM.addSample(fgdLabels.at<int>(i, 0), fgdSamples[i]);
	fgdGMM.UpdatePara();
}
//����ѭ����һ����Ϊÿ�����ط���GMM�������ĸ�˹ģ�ͣ�������partIndex�С�
static void assignGMMS(const Mat& _img, const Mat& _mask, const GMM& _bgdGMM, const GMM& _fgdGMM, Mat& _partIndex) {
	Point p;
	for (p.y = 0; p.y < _img.rows; p.y++) {
		for (p.x = 0; p.x < _img.cols; p.x++) {
			Vec3d color = (Vec3d)_img.at<Vec3b>(p);
			uchar t = _mask.at<uchar>(p);
			if (t == MUST_BGD || t == MAYBE_BGD)_partIndex.at<int>(p) = _bgdGMM.choice(color);
			else _partIndex.at<int>(p) = _fgdGMM.choice(color);
		}
	}
}
//����ѭ���ڶ��������ݵõ��Ľ������GMM����ֵ��
static void learnGMMs(const Mat& _img, const Mat& _mask, GMM& _bgdGMM, GMM& _fgdGMM, const Mat& _partIndex) {
	_bgdGMM.InitInterVar();
	_fgdGMM.InitInterVar();
	Point p;
	for (int i = 0; i < GMM::K; i++) {
		for (p.y = 0; p.y < _img.rows; p.y++) {
			for (p.x = 0; p.x < _img.cols; p.x++) {
				int tmp = _partIndex.at<int>(p);
				if (tmp == i) {
					if (_mask.at<uchar>(p) == MUST_BGD || _mask.at<uchar>(p) == MAYBE_BGD)
						_bgdGMM.addSample(tmp, _img.at<Vec3b>(p));
					else
						_fgdGMM.addSample(tmp, _img.at<Vec3b>(p));
				}
			}
		}
	}
	_bgdGMM.UpdatePara();
	_fgdGMM.UpdatePara();
}
//���ݵõ��Ľ������ͼ��ʹ�����̸����ֳɵĿ� Done
static void getGraph(const Mat& _img, const Mat& _mask, const GMM& _bgdGMM, const GMM& _fgdGMM, double _lambda, const Mat& _l, const Mat& _ul, const Mat& _u, const Mat& _ur, CutGraph& _graph) {
	int vCount = _img.cols*_img.rows;
	int eCount = 2 * (4 * vCount - 3 * _img.cols - 3 * _img.rows + 2);
	_graph = CutGraph(vCount, eCount);
	Point p;
	for (p.y = 0; p.y < _img.rows; p.y++) {
		for (p.x = 0; p.x < _img.cols; p.x++) {
			int vNum = _graph.addVertex();
			Vec3b color = _img.at<Vec3b>(p);
			double wSource = 0, wSink = 0;
			if (_mask.at<uchar>(p) == MAYBE_BGD || _mask.at<uchar>(p) == MAYBE_FGD) {
				wSource = -log(_bgdGMM.tWeight(color));
				wSink = -log(_fgdGMM.tWeight(color));
			}
			else if (_mask.at<uchar>(p) == MUST_BGD) wSink = _lambda;
			else wSource = _lambda;
			_graph.addVertexWeights(vNum, wSource, wSink);
			if (p.x > 0) {
				double w = _l.at<double>(p);
				_graph.addEdges(vNum, vNum - 1, w);
			}
			if (p.x > 0 && p.y > 0) {
				double w = _ul.at<double>(p);
				_graph.addEdges(vNum, vNum - _img.cols - 1, w);
			}
			if (p.y > 0) {
				double w = _u.at<double>(p);
				_graph.addEdges(vNum, vNum - _img.cols, w);
			}
			if (p.x < _img.cols - 1 && p.y > 0) {
				double w = _ur.at<double>(p);
				_graph.addEdges(vNum, vNum - _img.cols + 1, w);
			}
		}
	}
}
//���зָ� Done
static void estimateSegmentation(CutGraph& _graph, Mat& _mask) {
	_graph.maxFlow();
	Point p;
	for (p.y = 0; p.y < _mask.rows; p.y++) {
		for (p.x = 0; p.x < _mask.cols; p.x++) {
			if (_mask.at<uchar>(p) == MAYBE_BGD || _mask.at<uchar>(p) == MAYBE_FGD) {
				if (_graph.isSourceSegment(p.y*_mask.cols + p.x))
					_mask.at<uchar>(p) = MAYBE_FGD;
				else _mask.at<uchar>(p) = MAYBE_BGD;
			}
		}
	}
}
GrabCut2D::~GrabCut2D(void) {}

//***GrabCut ������
//һ.�������ͣ�
//���룺
//cv::InputArray _img,     :�����colorͼ��(����-cv:Mat)
//cv::Rect rect            :��ͼ���ϻ��ľ��ο�����-cv:Rect) 
//�м����
//cv::InputOutputArray _bgdModel ��   ����ģ�ͣ��Ƽ�GMM)������-13*n�������������double���͵��Զ������ݽṹ������Ϊcv:Mat������Vector/List/����ȣ�
//cv::InputOutputArray _fgdModel :    ǰ��ģ�ͣ��Ƽ�GMM) ������-13*n�������������double���͵��Զ������ݽṹ������Ϊcv:Mat������Vector/List/����ȣ�
//���:
//cv::InputOutputArray _mask  : ����ķָ��� (���ͣ� cv::Mat)
void GrabCut2D::GrabCut(InputArray _img, InputOutputArray _mask, Rect rect,
	InputOutputArray _bgdModel, InputOutputArray _fgdModel, int mode) {
	//����������ɫͼ��
	Mat img = _img.getMat();
	Mat& mask = _mask.getMatRef();
	Mat& bgdModel = _bgdModel.getMatRef();
	Mat& fgdModel = _fgdModel.getMatRef();
	//���岢��ʼ��GMM
	GMM bgdGMM(bgdModel), fgdGMM(fgdModel);
	//ǰ������ɫ���������о���(kmeans)
	//if (mode == GC_WITH_RECT)
	//Ϊʲô�����Ǿ����ʡ�ԣ���������
	initGMMs(img, mask, bgdGMM, fgdGMM);
	//����t-weight(�������n-weight��ƽ���
	//����ƽ���������ļ�����GMMģ����ʵ��
	const double gamma = 50;
	const double beta = calcuBeta(img);
	Mat leftW, upleftW, upW, uprightW;
	calcuNWeight(img, leftW, upleftW, upW, uprightW, beta, gamma);
	//����maxFlow����зָ�
	Mat compIdxs(img.size(), CV_32SC1);
	const double lambda = 9 * gamma;
	//���б��ֵĵ���
	CutGraph graph;
	assignGMMS(img, mask, bgdGMM, fgdGMM, compIdxs);//ƥ��GMMģ��
	learnGMMs(img, mask, bgdGMM, fgdGMM, compIdxs);//ѧϰGMM����
	getGraph(img, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW, graph);
	estimateSegmentation(graph, mask);//����max flow����Ԥ��ָ�
	//8.Save Result�������������mask�������mask��ǰ�������Ӧ�Ĳ�ɫͼ�񱣴����ʾ�ڽ��������У�
}