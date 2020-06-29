#pragma once
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>

class GMM {
public:
	//高斯模型的数量，按照论文中的实现，为5
	static const int K = 5;
	//GMM的构造函数，从 model 中读取参数并存储
	GMM(cv::Mat& _model);
	//计算某个颜色属于某个组件的可能性（高斯概率）
	double countPossi(int, const cv::Vec3d) const;
	//计算数据项权重
	double tWeight(const cv::Vec3d) const;
	//计算一个颜色应该是属于哪个组件（高斯概率最高的项）
	int judgeGMM(const cv::Vec3d) const;
	//学习之前对数据进行初始化
	void InitInterVar();
	//添加单个的点
	void addSample(int, const cv::Vec3d);
	//根据添加的数据，计算新的参数结果
	void UpdatePara();

	void outputGMM();

private:
	//计算协方差矩阵的逆和行列式的值
	void calDetAndInv(int);
	//存储GMM模型
	cv::Mat model;
	//每个GMM分量的 权重
	double *coefs;
	//每个GMM分量的 均值
	double *mean;
	//每个GMM分量的 协方差
	double *cov;
	//存储协方差的逆，便于计算
	double covInv[K][3][3];
	//存储协方差的行列式值
	double covDet[K];

	//用于学习过程中保存中间数据的变量
	double sums[K][3];//每个GMM模型的每个颜色通道的数值总和
	double prods[K][3][3];
	int sampleCounts[K];//每个GMM模型的像素点数
	int totalSampleCount;
};