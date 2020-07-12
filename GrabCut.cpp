#include "GrabCut.h"

const double myMax = 10000.0;
//���� Beta ��ֵ(���������еĹ�ʽ5)
//����Ħ���Ҫ����ʹ�ù�ʽ4�е�ָ���ڸ߶ԱȶȺ͵ͶԱȶ��±��ֵø���
double MyGrabCut::calBeta(const Mat& _img) {
	double totalDiff = 0;
	//����ȫͼ
	for (int y = 0; y < _img.rows; y++) {
		for (int x = 0; x < _img.cols; x++) {
			Vec3d color = (Vec3d)_img.at<Vec3b>(y, x);
			//�ֱ���㵱ǰ���������������ĸ�������ھӵ���ɫ��
			if (x > 0) {
				//������ͨ����ɫ��
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y, x - 1);
				//����ƽ��
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
	//��������ֵ����/�ܱ�������ע���ܱ���Ҫ���Ǳ߽������
	double expectation = totalDiff / (4 * _img.cols*_img.rows - 2 * _img.cols - 2 * _img.rows);
	double beta = 1.0 / (2 * expectation);//��ʽ5
	return beta;
}
//����ƽ����
void MyGrabCut::calcuNWeight(const Mat& _img, Mat& _l, Mat& _ul, Mat& _u, Mat& _ur, double _beta, double _gamma) {
	const double gammaDiv = _gamma / std::sqrt(2.0f);//б�ߵ�gammaֵ������2
	_l.create(_img.size(), CV_64FC1);
	_ul.create(_img.size(), CV_64FC1);
	_u.create(_img.size(), CV_64FC1);
	_ur.create(_img.size(), CV_64FC1);
	//���ڶԳ��ԣ��˸�������ֻ��Ҫ�����ĸ�����u=up,l=left,r=right��
	for (int y = 0; y < _img.rows; y++) {
		for (int x = 0; x < _img.cols; x++) {
			Vec3d color = (Vec3d)_img.at<Vec3b>(y, x);
			//������
			if (x - 1 >= 0) {
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y, x - 1);
				_l.at<double>(y, x) = _gamma*exp(-_beta*diff.dot(diff));//��ʽ4
			}
			else _l.at<double>(y, x) = 0;
			//�������Ϸ�
			if (x - 1 >= 0 && y - 1 >= 0) {
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y - 1, x - 1);
				_ul.at<double>(y, x) = gammaDiv*exp(-_beta*diff.dot(diff));
			}
			else _ul.at<double>(y, x) = 0;
			//�����Ϸ�
			if (y - 1 >= 0) {
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y - 1, x);
				_u.at<double>(y, x) = _gamma*exp(-_beta*diff.dot(diff));
			}
			else _u.at<double>(y, x) = 0;
			//�������Ϸ�
			if (x + 1 < _img.cols && y - 1 >= 0) {
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y - 1, x + 1);
				_ur.at<double>(y, x) = gammaDiv*exp(-_beta*diff.dot(diff));
			}
			else _ur.at<double>(y, x) = 0;
		}
	}
}

//����opencv�е� kmeans ������ʼ�� GMM ģ��
void MyGrabCut::initGMMs(const Mat& img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM) {
	const int maxTurn = 10;//���ѭ������
	Mat bgdLabels, fgdLabels;//���ڼ�¼kmeans����Ľ��
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
	//�Ա�����������kmeas���࣬��Ϊ5��
	Mat _bgdSamples((int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0]);
	//��������Ϊ���������ݣ��������������ǩ����ֹ��������ֹģʽ�����������������ȣ����ظ���������ʼ�����ĵ��㷨
	kmeans(_bgdSamples, GMM::K, bgdLabels,
		TermCriteria(CV_TERMCRIT_ITER, maxTurn, 0.0), 0, KMEANS_PP_CENTERS);
	//��ǰ����������kmeas���࣬��Ϊ5��
	Mat _fgdSamples((int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0]);
	kmeans(_fgdSamples, GMM::K, fgdLabels,
		TermCriteria(CV_TERMCRIT_ITER, maxTurn, 0.0), 0, KMEANS_PP_CENTERS);

	//��ʼ���м����
	bgdGMM.InitInterVar();
	//���ݾ������������ÿ��GMM����еľ�ֵ��Э����Ȳ���
	for (int i = 0; i < (int)bgdSamples.size(); i++)
		//����Ӧģ�����ӵ����㣬addSample��һ������Ϊ��Ӧģ�ͱ�ţ��ڶ�������Ϊ���ص���ɫ��Ϣ
		bgdGMM.addSample(bgdLabels.at<int>(i, 0), bgdSamples[i]);
	//����GMM����Ҫ����
	bgdGMM.UpdatePara();

	//��������
	fgdGMM.InitInterVar();
	for (int i = 0; i < (int)fgdSamples.size(); i++)
		fgdGMM.addSample(fgdLabels.at<int>(i, 0), fgdSamples[i]);
	fgdGMM.UpdatePara();
}

//����ѭ����һ����Ϊÿ�����ط���GMM�������ĸ�˹ģ�ͣ�������partIndex�С�
void MyGrabCut::assignGMM(const Mat& _img, const Mat& _mask, const GMM& _bgdGMM, const GMM& _fgdGMM, Mat& _GmmNumber) {
	Point p;
	//����ÿ������
	for (p.y = 0; p.y < _img.rows; p.y++) {
		for (p.x = 0; p.x < _img.cols; p.x++) {
			Vec3d color = (Vec3d)_img.at<Vec3b>(p);
			uchar t = _mask.at<uchar>(p);
			if (t == MUST_BGD || t == MAYBE_BGD) {//�ڱ���GMM��ѡ��
				_GmmNumber.at<int>(p) = _bgdGMM.judgeGMM(color);//Ϊ�õ�����������GMM����
			}
			else {//��ǰ��GMM��ѡ��
				_GmmNumber.at<int>(p) = _fgdGMM.judgeGMM(color);
			}
		}
	}
}

//����ѭ���ڶ��������ݵõ��Ľ������GMM����ֵ��
void MyGrabCut::learnGMMs(const Mat& _img, const Mat& _mask, GMM& _bgdGMM, GMM& _fgdGMM, const Mat& _GmmNumber) {
	_bgdGMM.InitInterVar();
	_fgdGMM.InitInterVar();
	Point p;
	//�ȷ�����������������������ĳ��GMM����������
	for (int i = 0; i < GMM::K; i++) {
		for (p.y = 0; p.y < _img.rows; p.y++) {
			for (p.x = 0; p.x < _img.cols; p.x++) {
				int tmp = _GmmNumber.at<int>(p);
				if (tmp == i) {
					if (_mask.at<uchar>(p) == MUST_BGD || _mask.at<uchar>(p) == MAYBE_BGD)
						_bgdGMM.addSample(tmp, _img.at<Vec3b>(p));
					else
						_fgdGMM.addSample(tmp, _img.at<Vec3b>(p));
				}
			}
		}
	}
	//�ٸ��²���
	_bgdGMM.UpdatePara();
	_fgdGMM.UpdatePara();
}

//���ݵõ��Ľ������ͼ�����ø������ӣ�
void MyGrabCut::getGraph(const Mat& _img, const Mat& _mask, const GMM& _bgdGMM, const GMM& _fgdGMM, const Mat& _l, const Mat& _ul, const Mat& _u, const Mat& _ur, GraphType *_graph) {
	Point p;
	//����ÿ����
	for (p.y = 0; p.y < _img.rows; p.y++) {
		for (p.x = 0; p.x < _img.cols; p.x++) {
			//���Ӷ��㲢��ȡ��ǰ�ڵ���
			int vNum = _graph->add_node();
			//��ȡ��ǰ�����ɫ
			Vec3b color = _img.at<Vec3b>(p);
			double wSource = 0;//����
			double wSink = 0;//���
			//����õ�Ĳ�β�ȷ��
			if (_mask.at<uchar>(p) == MAYBE_BGD || _mask.at<uchar>(p) == MAYBE_FGD) {
				//���ݻ�ϸ�˹�ܶ�ģ�ͷֱ�����Ⱥ����
				wSource = -log(_bgdGMM.tWeight(color));
				wSink = -log(_fgdGMM.tWeight(color));
			}
			//����õ�Ϊ�����㣬���Ϊ�������Ϊ0�����仰˵����Ȼ��ǰ������
			else if (_mask.at<uchar>(p) == MUST_BGD) wSink = myMax;
			//����õ�Ϊǰ���㣬����Ϊ������Ϊ0�����仰˵����Ȼ�ͱ�������
			else wSource = myMax;
			//Ϊ��ǰ�ڵ�������Ⱥͳ���ֵ
			_graph->add_tweights(vNum, wSource, wSink);
			//����ƽ����ı�
			if (p.x > 0) {//�����ұ�
				double weight = _l.at<double>(p);
				_graph->add_edge(vNum, vNum - 1, weight, weight);
			}
			if (p.x > 0 && p.y > 0) {//�����б�
				double weight = _ul.at<double>(p);
				_graph->add_edge(vNum, vNum - _img.cols - 1, weight, weight);
			}
			if (p.y > 0) {//�����б�
				double weight = _u.at<double>(p);
				_graph->add_edge(vNum, vNum - _img.cols, weight, weight);
			}
			if (p.x < _img.cols - 1 && p.y > 0) {//�����б�
				double weight = _ur.at<double>(p);
				_graph->add_edge(vNum, vNum - _img.cols + 1, weight, weight);
			}
		}
	}
}
//���зָ�
void MyGrabCut::estimateSegmentation(GraphType *_graph, Mat& _mask) {
	_graph->maxflow();//�������ӽ������������
	Point p;
	//����ÿ����
	for (p.y = 0; p.y < _mask.rows; p.y++) {
		for (p.x = 0; p.x < _mask.cols; p.x++) {
			if (_mask.at<uchar>(p) == MAYBE_BGD || _mask.at<uchar>(p) == MAYBE_FGD) {
				if (_graph->what_segment(p.y*_mask.cols + p.x) == Graph<double, double, double>::SOURCE)//����ָ����Ŀ��
					_mask.at<uchar>(p) = MAYBE_FGD;//��ֵΪ����ǰ��
				else _mask.at<uchar>(p) = MAYBE_BGD;//��ֵΪ���ܱ���
			}
		}
	}
}
MyGrabCut::~MyGrabCut(void) {}

//GrabCut������
void MyGrabCut::GrabCut(InputArray _img, InputOutputArray _mask, Rect rect,
	InputOutputArray _bgdModel, InputOutputArray _fgdModel, int mode) {

	//����������ɫͼ��
	Mat img = _img.getMat();//copy
	Mat& mask = _mask.getMatRef();//��ַ
	Mat& bgdModel = _bgdModel.getMatRef();//��ַ
	Mat& fgdModel = _fgdModel.getMatRef();//��ַ
	//��ʼ�� ���� ��ȡ�ղż���� GMMģ��
	GMM bgdGMM(bgdModel), fgdGMM(fgdModel);
	//����ǵ�һ�ε�������Ҫ����kmeans����GMM�����ľ������
	if (mode == GC_WITH_RECT)
		initGMMs(img, mask, bgdGMM, fgdGMM);
	//�������GMMģ�͵Ĳ���
	//cout << "----------------Background Model-----------------"<<endl;
	//bgdGMM.outputGMM();
	//cout << "----------------Foreground Model-----------------"<<endl;
	//fgdGMM.outputGMM();

	//GmmNumber���ڼ�¼ÿ������������GMM�����ı��
	Mat GmmNumber(img.size(), CV_32SC1);

	//���б��ֵĵ�����Inerative Minimisation��
	assignGMM(img, mask, bgdGMM, fgdGMM, GmmNumber);//1.ƥ��GMMģ��
	learnGMMs(img, mask, bgdGMM, fgdGMM, GmmNumber);//2.ѧϰGMM����
	//��¼ƽ����
	Mat leftW, upleftW, upW, uprightW;
	const double gamma = 50;//gammaΪ����ֵ�������������ᵽ
	const double beta = calBeta(img);//���ݹ�ʽ5����betaֵ
	//����neighbor-weightƽ����(�����ϡ��ϡ�����)
	calcuNWeight(img, leftW, upleftW, upW, uprightW, beta, gamma);
	//��ȡ���������
	int vertexCount = img.cols*img.rows;
	int edgeCount = (4 * vertexCount - 2 * img.cols - 2 * img.rows) / 2 + 1;
	//��ʼ��CutGraph����
	GraphType graph = GraphType(vertexCount, edgeCount);
	getGraph(img, mask, bgdGMM, fgdGMM, leftW, upleftW, upW, uprightW, &graph);//3.����ͼgraph������terminal-weight�����Ϊ�ָ���׼��
	estimateSegmentation(&graph, mask);//4.����max flow��ͼgraph����Ԥ��ָ�
}