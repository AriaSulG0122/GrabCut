#pragma once
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>

class GMM {
public:
	//��˹ģ�͵����������������е�ʵ�֣�Ϊ5
	static const int K = 5;
	//GMM�Ĺ��캯������ model �ж�ȡ�������洢
	GMM(cv::Mat& _model);
	//����ĳ����ɫ����ĳ������Ŀ����ԣ���˹���ʣ�
	double countPossi(int, const cv::Vec3d) const;
	//����������Ȩ��
	double tWeight(const cv::Vec3d) const;
	//����һ����ɫӦ���������ĸ��������˹������ߵ��
	int judgeGMM(const cv::Vec3d) const;
	//ѧϰ֮ǰ�����ݽ��г�ʼ��
	void InitInterVar();
	//��ӵ����ĵ�
	void addSample(int, const cv::Vec3d);
	//������ӵ����ݣ������µĲ������
	void UpdatePara();

	void outputGMM();

private:
	//����Э���������������ʽ��ֵ
	void calDetAndInv(int);
	//�洢GMMģ��
	cv::Mat model;
	//ÿ��GMM������ Ȩ��
	double *coefs;
	//ÿ��GMM������ ��ֵ
	double *mean;
	//ÿ��GMM������ Э����
	double *cov;
	//�洢Э������棬���ڼ���
	double covInv[K][3][3];
	//�洢Э���������ʽֵ
	double covDet[K];

	//����ѧϰ�����б����м����ݵı���
	double sums[K][3];//ÿ��GMMģ�͵�ÿ����ɫͨ������ֵ�ܺ�
	double prods[K][3][3];
	int sampleCounts[K];//ÿ��GMMģ�͵����ص���
	int totalSampleCount;
};