#include "GMM.h"
#include <vector>
using namespace std;
using namespace cv;

//GMM的构造函数，从 model 中读取参数并存储
GMM::GMM(Mat& _model) {
	//GMM模型有13*K项数据，一个权重，三个均值和九个协方差
	//如果模型为空，则创建一个新的
	if (_model.empty()) {
		_model.create(1, 13 * K, CV_64FC1);
		_model.setTo(Scalar(0));
	}
	model = _model;
	//存储顺序为权重、均值和协方差
	coefs = model.ptr<double>(0);//占一个位
	mean = coefs + K;//占3K个位
	cov = mean + 3 * K;//占9K个位
	//如果某个项的权重不为0，则计算其协方差的逆和行列式
	for (int i = 0; i < K; i++)
		if (coefs[i] > 0)
			calDetAndInv(i);
}
//按照混合高斯密度模型进行计算高斯概率
double GMM::countPossi(int _i, const Vec3d _color) const {
	double res = 0;
	if (coefs[_i] > 0) {
		Vec3d diff = _color;
		double* curMean = mean + 3 * _i;
		//得到各原色的x-u值
		diff[0] -= curMean[0]; diff[1] -= curMean[1]; diff[2] -= curMean[2];
		//计算(x-u)^T * Sigma-1(x-u)
		double mult = diff[0] * (diff[0] * covInv[_i][0][0] + diff[1] * covInv[_i][1][0] + diff[2] * covInv[_i][2][0])
			+ diff[1] * (diff[0] * covInv[_i][0][1] + diff[1] * covInv[_i][1][1] + diff[2] * covInv[_i][2][1])
			+ diff[2] * (diff[0] * covInv[_i][0][2] + diff[1] * covInv[_i][1][2] + diff[2] * covInv[_i][2][2]);
		res = 1.0f / sqrt(covDet[_i]) * exp(-0.5f*mult);
	}
	return res;
}
//计算一个像素的数据项的值D(x)，根据混合高斯密度模型来
double GMM::tWeight(const Vec3d _color)const {
	double res = 0;
	for (int ci = 0; ci < K; ci++)
		res += coefs[ci] * countPossi(ci, _color);
	return res;
}

//计算一个三通道颜色属于哪个GMM
int GMM::judgeGMM(const Vec3d _color) const {
	int k = 0;
	double max = 0;
	//分别计算高斯概率，并分配给概率最大的GMM
	for (int i = 0; i < K; i++) {
		double p = countPossi(i, _color);
		if (p > max) {
			k = i;
			max = p;
		}
	}
	return k;
}

//学习之前对数据进行初始化
void GMM::InitInterVar() {
	//对要用的中间变量赋0
	for (int i = 0; i < K; i++) {
		for (int j = 0; j < 3; j++)
			sums[i][j] = 0;
		for (int p = 0; p < 3; p++) {
			for (int q = 0; q < 3; q++) {
				prods[i][p][q] = 0;
			}
		}
		sampleCounts[i] = 0;
	}
	totalSampleCount = 0;
}
//添加单个的点
void GMM::addSample(int _i, const Vec3d _color) {
	//改变中间变量的值
	for (int i = 0; i < 3; i++) {
		//增加颜色总和
		sums[_i][i] += _color[i];
		//刷新i行j列的值，为计算协方差做准备
		for (int j = 0; j < 3; j++)
			prods[_i][i][j] += _color[i] * _color[j];
	}
	sampleCounts[_i]++;//该模型像素加一
	totalSampleCount++;//总体样本数+1
}

//根据添加的数据，计算新的参数结果，前景和背景分开处理
void GMM::UpdatePara() {
	const double variance = 0.01;
	//遍历每个GMM模型分量
	for (int i = 0; i < K; i++) {
		int curCount = sampleCounts[i];//当前模型分量的像素数
		if (curCount == 0)	coefs[i] = 0;
		else {
			//计算高斯模型新的参数
			//权重
			coefs[i] = 1.0 * curCount / totalSampleCount;
			//先获取当前的均值起始位置
			double * curMean = mean + 3 * i;
			//计算当前列（某个原色）的均值
			for (int j = 0; j < 3; j++) {
				curMean[j] = sums[i][j] / curCount;
			}
			//先获取当前协方差的起始位置
			double* curCov = cov + 9 * i;
			//计算协方差
			for (int p = 0; p < 3; p++) {
				for (int q = 0; q < 3; q++) {
					//计算p行q列位置上的协方差
					curCov[p * 3 + q] = prods[i][p][q] / curCount - curMean[p] * curMean[q];
				}
			}
			//计算行列式
			double dtrm = curCov[0] * (curCov[4] * curCov[8] - curCov[5] * curCov[7]) - curCov[1] * (curCov[3] * curCov[8] - curCov[5] * curCov[6]) + curCov[2] * (curCov[3] * curCov[7] - curCov[4] * curCov[6]);
			//如果行列式值太小，则加入一些噪音，避免误差扩散
			if (dtrm <= std::numeric_limits<double>::epsilon()) {
				curCov[0] += variance;
				curCov[4] += variance;
				curCov[8] += variance;
			}
			//计算协方差的 行列式 和 逆
			calDetAndInv(i);
		}
	}
}
//计算协方差矩阵的 行列式 和 逆 的值
void GMM::calDetAndInv(int _i) {
	if (coefs[_i] > 0) {
		double *c = cov + 9 * _i;
		//行列式的值
		double dtrm = covDet[_i] = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6]);
		//计算逆矩阵，根据伴随矩阵求逆矩阵
		covInv[_i][0][0] = (c[4] * c[8] - c[5] * c[7]) / dtrm;
		covInv[_i][1][0] = -(c[3] * c[8] - c[5] * c[6]) / dtrm;
		covInv[_i][2][0] = (c[3] * c[7] - c[4] * c[6]) / dtrm;
		covInv[_i][0][1] = -(c[1] * c[8] - c[2] * c[7]) / dtrm;
		covInv[_i][1][1] = (c[0] * c[8] - c[2] * c[6]) / dtrm;
		covInv[_i][2][1] = -(c[0] * c[7] - c[1] * c[6]) / dtrm;
		covInv[_i][0][2] = (c[1] * c[5] - c[2] * c[4]) / dtrm;
		covInv[_i][1][2] = -(c[0] * c[5] - c[2] * c[3]) / dtrm;
		covInv[_i][2][2] = (c[0] * c[4] - c[1] * c[3]) / dtrm;
	}
}

void GMM::outputGMM() {
	for (int i = 0; i < K; i++) {
		cout << "Model:" << i << endl;
		cout << "coefs:" << coefs[i] << endl;
		cout << "mean:" << mean[i * 3 + 0] << " " << mean[i * 3 + 1] << " " << mean[i * 3 + 2] << endl;
		/*
		cout << "cov:" << endl;
		for (int j = 0; j < 3; j++) {
			cout << "\t" << cov[9 * i + j * 3] << "\t\t" << cov[9 * i + j * 3 + 1] << "\t\t" << cov[9 * i + j * 3 + 2] << endl;
		}
		*/
	}
}