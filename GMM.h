#ifndef GMM_H_
#define GMM_H_
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
//��˹����ģ�ͣ���Ҫ���� BorderMatting ������
class Gauss {
public:
	//��˹ģ�͵Ĺ��캯��
	Gauss();
	//�����˹���ʡ�
	static double gauss(const double, const double, const double);
	//�����˹ģ�͵ĸ���(����һ�ַ�ʽ��
	static double possibility(const cv::Vec3f&, const cv::Mat&, cv::Vec3f);
	//����������������ɢ������
	static void discret(std::vector<double>&, std::vector<double>&);//delta range from [0,6], sigma range from [0,delta/3]
	//���˹ģ���м���һ������
	void addsample(cv::Vec3f);
	//��������ģ�ͣ������˹ģ���еľ�ֵ��Э����
	void learn();

	cv::Vec3f getmean()const { return mean; }
	cv::Mat getcovmat()const { return covmat; }
private:
	//��˹ģ�͵ľ�ֵ
	cv::Vec3f mean;
	//��˹ģ�͵�Э����
	cv::Mat covmat;
	//��������
	std::vector<cv::Vec3f> samples;
	
};


class GMM {
public:
	//��˹ģ�͵����������������е�ʵ�֣�Ϊ5
	static const int K = 5;
	//GMM�Ĺ��캯������ model �ж�ȡ�������洢
	GMM(cv::Mat& _model);
	//����ĳ����ɫ����ĳ������Ŀ����ԣ���˹���ʣ�
	double possibility(int, const cv::Vec3d) const;
	//����������Ȩ��
	double tWeight(const cv::Vec3d) const;
	//����һ����ɫӦ���������ĸ��������˹������ߵ��
	int choice(const cv::Vec3d) const;
	//ѧϰ֮ǰ�����ݽ��г�ʼ��
	void InitInterVar();
	//���ӵ����ĵ�
	void addSample(int, const cv::Vec3d);
	//�������ӵ����ݣ������µĲ������
	void UpdatePara();
private:
	//����Э���������������ʽ��ֵ
	void calcuInvAndDet(int);
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
#endif