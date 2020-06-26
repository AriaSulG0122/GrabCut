#ifndef GMM_H_
#define GMM_H_
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
//高斯概率模型，主要用于 BorderMatting 过程中
class Gauss {
public:
	//高斯模型的构造函数
	Gauss();
	//计算高斯概率。
	static double gauss(const double, const double, const double);
	//计算高斯模型的概率(另外一种方式）
	static double possibility(const cv::Vec3f&, const cv::Mat&, cv::Vec3f);
	//对两个参数进行离散化处理
	static void discret(std::vector<double>&, std::vector<double>&);//delta range from [0,6], sigma range from [0,delta/3]
	//向高斯模型中加入一个样例
	void addsample(cv::Vec3f);
	//根据样例模型，计算高斯模型中的均值和协方差
	void learn();
	cv::Vec3f getmean()const { return mean; }
	cv::Mat getcovmat()const { return covmat; }
private:
	//高斯模型的均值
	cv::Vec3f mean;
	//高斯模型的协方差
	cv::Mat covmat;
	//样例集合
	std::vector<cv::Vec3f> samples;
	
};


class GMM {
public:
	//高斯模型的数量，按照论文中的实现，为5
	static const int K = 5;
	//GMM的构造函数，从 model 中读取参数并存储
	GMM(cv::Mat& _model);
	//计算某个颜色属于某个组件的可能性（高斯概率）
	double possibility(int, const cv::Vec3d) const;
	//计算数据项权重
	double tWeight(const cv::Vec3d) const;
	//计算一个颜色应该是属于哪个组件（高斯概率最高的项）
	int choice(const cv::Vec3d) const;
	//学习之前对数据进行初始化
	void learningBegin();
	//添加单个的点
	void addSample(int, const cv::Vec3d);
	//根据添加的数据，计算新的参数结果
	void learningEnd();
private:
	//计算协方差矩阵的逆和行列式的值
	void calcuInvAndDet(int);
	//存储GMM模型
	cv::Mat model;
	//GMM模型中，每个高斯分布的权重、均值和协方差
	double *coefs, *mean, *cov;
	//存储协方差的逆，便于计算
	double covInv[K][3][3];
	//存储协方差的行列式值
	double covDet[K];
	//用于学习过程中保存中间数据的变量
	double sums[K][3];
	double prods[K][3][3];
	int sampleCounts[K];
	int totalSampleCount;
};
#endif
