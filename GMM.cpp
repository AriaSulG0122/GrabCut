#include "GMM.h"
#include <vector>
using namespace std;
using namespace cv;

//GMM�Ĺ��캯������ model �ж�ȡ�������洢
GMM::GMM(Mat& _model) {
	//GMMģ����13*K�����ݣ�һ��Ȩ�أ�������ֵ�;Ÿ�Э����
	//���ģ��Ϊ�գ��򴴽�һ���µ�
	if (_model.empty()) {
		_model.create(1, 13 * K, CV_64FC1);
		_model.setTo(Scalar(0));
	}
	model = _model;
	//�洢˳��ΪȨ�ء���ֵ��Э����
	coefs = model.ptr<double>(0);//ռһ��λ
	mean = coefs + K;//ռ3K��λ
	cov = mean + 3 * K;//ռ9K��λ
	//���ĳ�����Ȩ�ز�Ϊ0���������Э������������ʽ
	for (int i = 0; i < K; i++)
		if (coefs[i] > 0)
			calDetAndInv(i);
}
//���ջ�ϸ�˹�ܶ�ģ�ͽ��м����˹����
double GMM::countPossi(int _i, const Vec3d _color) const {
	double res = 0;
	if (coefs[_i] > 0) {
		Vec3d diff = _color;
		double* curMean = mean + 3 * _i;
		//�õ���ԭɫ��x-uֵ
		diff[0] -= curMean[0]; diff[1] -= curMean[1]; diff[2] -= curMean[2];
		//����(x-u)^T * Sigma-1(x-u)
		double mult = diff[0] * (diff[0] * covInv[_i][0][0] + diff[1] * covInv[_i][1][0] + diff[2] * covInv[_i][2][0])
			+ diff[1] * (diff[0] * covInv[_i][0][1] + diff[1] * covInv[_i][1][1] + diff[2] * covInv[_i][2][1])
			+ diff[2] * (diff[0] * covInv[_i][0][2] + diff[1] * covInv[_i][1][2] + diff[2] * covInv[_i][2][2]);
		res = 1.0f / sqrt(covDet[_i]) * exp(-0.5f*mult);
	}
	return res;
}
//����һ�����ص��������ֵD(x)�����ݻ�ϸ�˹�ܶ�ģ����
double GMM::tWeight(const Vec3d _color)const {
	double res = 0;
	for (int ci = 0; ci < K; ci++)
		res += coefs[ci] * countPossi(ci, _color);
	return res;
}

//����һ����ͨ����ɫ�����ĸ�GMM
int GMM::judgeGMM(const Vec3d _color) const {
	int k = 0;
	double max = 0;
	//�ֱ�����˹���ʣ����������������GMM
	for (int i = 0; i < K; i++) {
		double p = countPossi(i, _color);
		if (p > max) {
			k = i;
			max = p;
		}
	}
	return k;
}

//ѧϰ֮ǰ�����ݽ��г�ʼ��
void GMM::InitInterVar() {
	//��Ҫ�õ��м������0
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
//��ӵ����ĵ�
void GMM::addSample(int _i, const Vec3d _color) {
	//�ı��м������ֵ
	for (int i = 0; i < 3; i++) {
		//������ɫ�ܺ�
		sums[_i][i] += _color[i];
		//ˢ��i��j�е�ֵ��Ϊ����Э������׼��
		for (int j = 0; j < 3; j++)
			prods[_i][i][j] += _color[i] * _color[j];
	}
	sampleCounts[_i]++;//��ģ�����ؼ�һ
	totalSampleCount++;//����������+1
}

//������ӵ����ݣ������µĲ��������ǰ���ͱ����ֿ�����
void GMM::UpdatePara() {
	const double variance = 0.01;
	//����ÿ��GMMģ�ͷ���
	for (int i = 0; i < K; i++) {
		int curCount = sampleCounts[i];//��ǰģ�ͷ�����������
		if (curCount == 0)	coefs[i] = 0;
		else {
			//�����˹ģ���µĲ���
			//Ȩ��
			coefs[i] = 1.0 * curCount / totalSampleCount;
			//�Ȼ�ȡ��ǰ�ľ�ֵ��ʼλ��
			double * curMean = mean + 3 * i;
			//���㵱ǰ�У�ĳ��ԭɫ���ľ�ֵ
			for (int j = 0; j < 3; j++) {
				curMean[j] = sums[i][j] / curCount;
			}
			//�Ȼ�ȡ��ǰЭ�������ʼλ��
			double* curCov = cov + 9 * i;
			//����Э����
			for (int p = 0; p < 3; p++) {
				for (int q = 0; q < 3; q++) {
					//����p��q��λ���ϵ�Э����
					curCov[p * 3 + q] = prods[i][p][q] / curCount - curMean[p] * curMean[q];
				}
			}
			//��������ʽ
			double dtrm = curCov[0] * (curCov[4] * curCov[8] - curCov[5] * curCov[7]) - curCov[1] * (curCov[3] * curCov[8] - curCov[5] * curCov[6]) + curCov[2] * (curCov[3] * curCov[7] - curCov[4] * curCov[6]);
			//�������ʽֵ̫С�������һЩ���������������ɢ
			if (dtrm <= std::numeric_limits<double>::epsilon()) {
				curCov[0] += variance;
				curCov[4] += variance;
				curCov[8] += variance;
			}
			//����Э����� ����ʽ �� ��
			calDetAndInv(i);
		}
	}
}
//����Э�������� ����ʽ �� �� ��ֵ
void GMM::calDetAndInv(int _i) {
	if (coefs[_i] > 0) {
		double *c = cov + 9 * _i;
		//����ʽ��ֵ
		double dtrm = covDet[_i] = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6]);
		//��������󣬸��ݰ�������������
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