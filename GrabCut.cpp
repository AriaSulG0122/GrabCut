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

//计算 Beta 的值(根据论文中的公式5)
//这里的β主要用于使得公式4中的指数在高对比度和低对比度下表现得更好
static double calBeta(const Mat& _img) {
	double totalDiff = 0;
	//遍历全图
	for (int y = 0; y < _img.rows; y++) {
		for (int x = 0; x < _img.cols; x++) {
			Vec3d color = (Vec3d)_img.at<Vec3b>(y, x);
			//分别计算当前像素与上下左右四个方向的邻居的颜色差
			if (x > 0) {
				//计算三通道的色差
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y, x - 1);
				//计算平方
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
	//计算期望值（和/总边数），注意总边数要考虑边界点的情况
	double expectation = totalDiff / (4 * _img.cols*_img.rows - 2 * _img.cols - 2 * _img.rows);
	double beta = 1.0 / (2 * expectation);//公式5
	return beta;
}
//计算平滑项
static void calcuNWeight(const Mat& _img, Mat& _l, Mat& _ul, Mat& _u, Mat& _ur, double _beta, double _gamma) {
	const double gammaDiv = _gamma / std::sqrt(2.0f);//斜边的gamma值除根号2
	_l.create(_img.size(), CV_64FC1);
	_ul.create(_img.size(), CV_64FC1);
	_u.create(_img.size(), CV_64FC1);
	_ur.create(_img.size(), CV_64FC1);
	//由于对称性，八个点我们只需要计算四个方（u=up,l=left,r=right）
	for (int y = 0; y < _img.rows; y++) {
		for (int x = 0; x < _img.cols; x++) {
			Vec3d color = (Vec3d)_img.at<Vec3b>(y, x);
			//计算左方
			if (x - 1 >= 0) {
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y, x - 1);
				_l.at<double>(y, x) = _gamma*exp(-_beta*diff.dot(diff));//公式4
			}
			else _l.at<double>(y, x) = 0;
			//计算左上方
			if (x - 1 >= 0 && y - 1 >= 0) {
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y - 1, x - 1);
				_ul.at<double>(y, x) = gammaDiv*exp(-_beta*diff.dot(diff));
			}
			else _ul.at<double>(y, x) = 0;
			//计算上方
			if (y - 1 >= 0) {
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y - 1, x);
				_u.at<double>(y, x) = _gamma*exp(-_beta*diff.dot(diff));
			}
			else _u.at<double>(y, x) = 0;
			//计算右上方
			if (x + 1 < _img.cols && y - 1 >= 0) {
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y - 1, x + 1);
				_ur.at<double>(y, x) = gammaDiv*exp(-_beta*diff.dot(diff));
			}
			else _ur.at<double>(y, x) = 0;
		}
	}
}

//利用opencv中的 kmeans 方法初始化 GMM 模型
static void initGMMs(const Mat& img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM) {
	const int maxTurn = 10;//最多循环次数
	Mat bgdLabels, fgdLabels;//用于记录kmeans分类的结果
	vector<Vec3f> bgdSamples, fgdSamples;//Vec3f为三通道float，在这里记录了RGB三通道信息
	Point p;
	for (p.y = 0; p.y < img.rows; p.y++) {
		for (p.x = 0; p.x < img.cols; p.x++) {
			if (mask.at<uchar>(p) == MUST_BGD || mask.at<uchar>(p) == MAYBE_BGD)
				bgdSamples.push_back((Vec3f)img.at<Vec3b>(p));//推入背景样本
			else
				fgdSamples.push_back((Vec3f)img.at<Vec3b>(p));//推入前景样本
		}
	}
	//对背景样本进行kmeas聚类，分为5类
	Mat _bgdSamples((int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0]);
	//参数依次为：输入数据，聚类数，聚类标签，终止条件（终止模式、最大迭代次数、精度），重复次数，初始化中心的算法
	kmeans(_bgdSamples, GMM::K, bgdLabels,
		TermCriteria(CV_TERMCRIT_ITER, maxTurn, 0.0), 0, KMEANS_PP_CENTERS);
	//对前景样本进行kmeas聚类，分为5类
	Mat _fgdSamples((int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0]);
	kmeans(_fgdSamples, GMM::K, fgdLabels,
		TermCriteria(CV_TERMCRIT_ITER, maxTurn, 0.0), 0, KMEANS_PP_CENTERS);

	//初始化中间变量
	bgdGMM.InitInterVar();
	//根据聚类的样本更新每个GMM组件中的均值、协方差等参数
	for (int i = 0; i < (int)bgdSamples.size(); i++)
		//往对应模型增加单个点，addSample第一个参数为对应模型编号，第二个参数为像素点颜色信息
		bgdGMM.addSample(bgdLabels.at<int>(i, 0), bgdSamples[i]);
	//更新GMM的主要参数
	bgdGMM.UpdatePara();

	//背景类似
	fgdGMM.InitInterVar();
	for (int i = 0; i < (int)fgdSamples.size(); i++)
		fgdGMM.addSample(fgdLabels.at<int>(i, 0), fgdSamples[i]);
	fgdGMM.UpdatePara();
}

//迭代循环第一步，为每个像素分配GMM中所属的高斯模型，保存在partIndex中。
static void assignGMM(const Mat& _img, const Mat& _mask, const GMM& _bgdGMM, const GMM& _fgdGMM, Mat& _partIndex) {
	Point p;
	//遍历每个像素
	for (p.y = 0; p.y < _img.rows; p.y++) {
		for (p.x = 0; p.x < _img.cols; p.x++) {
			Vec3d color = (Vec3d)_img.at<Vec3b>(p);
			uchar t = _mask.at<uchar>(p);
			if (t == MUST_BGD || t == MAYBE_BGD)_partIndex.at<int>(p) = _bgdGMM.judgeGMM(color);//在背景GMM中选择
			else _partIndex.at<int>(p) = _fgdGMM.judgeGMM(color);//在前景GMM中选择
		}
	}
}

//迭代循环第二步，根据得到的结果计算GMM参数值。
static void learnGMMs(const Mat& _img, const Mat& _mask, GMM& _bgdGMM, GMM& _fgdGMM, const Mat& _partIndex) {
	_bgdGMM.InitInterVar();
	_fgdGMM.InitInterVar();
	Point p;
	//先分配样本
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
	//再更新参数
	_bgdGMM.UpdatePara();
	_fgdGMM.UpdatePara();
}

//根据得到的结果构造图（调用给的轮子）
static void getGraph(const Mat& _img, const Mat& _mask, const GMM& _bgdGMM, const GMM& _fgdGMM, double myMax, const Mat& _l, const Mat& _ul, const Mat& _u, const Mat& _ur, CutGraph& _graph) {
	//获取点数与边数
	int vertexCount = _img.cols*_img.rows;
	int edgeCount = (4 * vertexCount - 2 * _img.cols - 2 * _img.rows) / 2 + 1;
	//初始化CutGraph对象
	_graph = CutGraph(vertexCount, edgeCount);
	Point p;
	//遍历每个点
	for (p.y = 0; p.y < _img.rows; p.y++) {
		for (p.x = 0; p.x < _img.cols; p.x++) {
			//增加顶点并获取当前节点编号
			int vNum = _graph.addVertex();
			Vec3b color = _img.at<Vec3b>(p);
			double wSource = 0;//出度
			double wSink = 0;//入度
			//如果该点的层次不确定
			if (_mask.at<uchar>(p) == MAYBE_BGD || _mask.at<uchar>(p) == MAYBE_FGD) {
				//根据混合高斯密度模型分别算出度和入度
				wSource = -log(_bgdGMM.tWeight(color));
				wSink = -log(_fgdGMM.tWeight(color));
			}
			//如果该点为背景点，入度为无穷，即min cut分割时不会去切这条边
			else if (_mask.at<uchar>(p) == MUST_BGD) wSink = myMax;
			//如果该点为前景点，同理
			else wSource = myMax;
			//为当前节点增加入度和出度值
			_graph.addVertexWeights(vNum, wSource, wSink);
			//增加平滑项的边
			if (p.x > 0) {
				double weight = _l.at<double>(p);
				_graph.addEdges(vNum, vNum - 1, weight);
			}
			if (p.x > 0 && p.y > 0) {
				double weight = _ul.at<double>(p);
				_graph.addEdges(vNum, vNum - _img.cols - 1, weight);
			}
			if (p.y > 0) {
				double weight = _u.at<double>(p);
				_graph.addEdges(vNum, vNum - _img.cols, weight);
			}
			if (p.x < _img.cols - 1 && p.y > 0) {
				double weight = _ur.at<double>(p);
				_graph.addEdges(vNum, vNum - _img.cols + 1, weight);
			}
		}
	}
}
//进行分割
static void estimateSegmentation(CutGraph& _graph, Mat& _mask) {
	_graph.maxFlow();//调用轮子进行最大流计算
	Point p;
	//遍历每个点
	for (p.y = 0; p.y < _mask.rows; p.y++) {
		for (p.x = 0; p.x < _mask.cols; p.x++) {
			if (_mask.at<uchar>(p) == MAYBE_BGD || _mask.at<uchar>(p) == MAYBE_FGD) {
				if (_graph.isSourceSegment(p.y*_mask.cols + p.x))//如果分割后是目标
					_mask.at<uchar>(p) = MAYBE_FGD;//赋值为可能前景
				else _mask.at<uchar>(p) = MAYBE_BGD;//赋值为可能背景
			}
		}
	}
}
GrabCut2D::~GrabCut2D(void) {}

//GrabCut主函数
void GrabCut2D::GrabCut(InputArray _img, InputOutputArray _mask, Rect rect,
	InputOutputArray _bgdModel, InputOutputArray _fgdModel, int mode) {
	//加载输入颜色图像
	Mat img = _img.getMat();
	Mat& mask = _mask.getMatRef();
	Mat& bgdModel = _bgdModel.getMatRef();
	Mat& fgdModel = _fgdModel.getMatRef();
	//初始化或者获取GMM模型
	GMM bgdGMM(bgdModel), fgdGMM(fgdModel);
	bgdGMM.outputGMM();
	//如果是第一次迭代，需要利用kmeans进行GMM分量的聚类操作
	if (mode == GC_WITH_RECT) {
		initGMMs(img, mask, bgdGMM, fgdGMM);
	}
	bgdGMM.outputGMM();
	//计算terminal-weight(数据项）和neighbor-weight（平滑项）
	const double gamma = 50;//gamma为经验值，在论文中有提到
	const double beta = calBeta(img);//根据公式5计算beta值
	Mat leftW, upleftW, upW, uprightW;
	//计算平滑项(左、左上、上、右上)
	calcuNWeight(img, leftW, upleftW, upW, uprightW, beta, gamma);
	//compIdxs用于记录迭代过程中的数据
	Mat compIdxs(img.size(), CV_32SC1);
	const double myMax = 10000;
	CutGraph graph;
	//进行本轮的迭代（Inerative Minimisation）
	assignGMM(img, mask, bgdGMM, fgdGMM, compIdxs);//匹配GMM模型
	learnGMMs(img, mask, bgdGMM, fgdGMM, compIdxs);//学习GMM参数
	getGraph(img, mask, bgdGMM, fgdGMM, myMax, leftW, upleftW, upW, uprightW, graph);//创建图graph，为分割做准备
	estimateSegmentation(graph, mask);//调用max flow对图graph进行预测分割
}