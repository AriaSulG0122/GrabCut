#include <vector>
#include "GCApplication.h"

//恢复初始状态
void GCApplication::reset()
{
	if (!mask.empty())
		mask.setTo(Scalar::all(GC_BGD));
	bgdPxls.clear(); fgdPxls.clear();
	prBgdPxls.clear();  prFgdPxls.clear();

	isInitialized = false;
	rectState = NOT_SET;
	lblsState = NOT_SET;
	prLblsState = NOT_SET;
	iterCount = 0;
}

//设置图片与窗口名称
void GCApplication::setImageAndWinName(const Mat& _image, const string& _winName)
{
	if (_image.empty() || _winName.empty())
		return;
	image = &_image;
	winName = &_winName;
	mask.create(image->size(), CV_8UC1);
	reset();
}

//Copy the value of comMask to binMask
void GCApplication::getBinMask( const Mat& comMask, Mat& binMask )
{
	if( comMask.empty() || comMask.type()!=CV_8UC1 )
		CV_Error( CV_StsBadArg, "comMask is empty or has incorrect type (not CV_8UC1)" );
	if( binMask.empty() || binMask.rows!=comMask.rows || binMask.cols!=comMask.cols )
		binMask.create( comMask.size(), CV_8UC1 );
	binMask = comMask & 1;//令MUST_BGD和MAYBE_BGD变为0
}


//显示图片与用户交互信息
void GCApplication::showImage()
{
	if (image->empty() || winName->empty())
		return;

	Mat res;
	Mat binMask;
	if (!isInitialized)
		image->copyTo(res);
	else
	{
		getBinMask(mask, binMask);
		
		image->copyTo(res, binMask); //mask为0的地方会被遮罩，其余部分正常显示
	}
	imwrite("result.jpg", res);
	vector<Point>::const_iterator it;
	//根据用户的交互信息，读取不同的像素容器并展示到图片上
	for (it = bgdPxls.begin(); it != bgdPxls.end(); ++it)
		circle(res, *it, radius, BLUE, thickness);
	for (it = fgdPxls.begin(); it != fgdPxls.end(); ++it)
		circle(res, *it, radius, GREEN, thickness);
	for (it = prBgdPxls.begin(); it != prBgdPxls.end(); ++it)
		circle(res, *it, radius, LIGHTBLUE, thickness);
	for (it = prFgdPxls.begin(); it != prFgdPxls.end(); ++it)
		circle(res, *it, radius, PINK, thickness);

	//绘制选中框
	if (rectState == IN_PROCESS || rectState == SET)
		rectangle(res, Point(rect.x, rect.y), Point(rect.x + rect.width, rect.y + rect.height), RED, 2);
	//展示分割后的图像
	imshow(*winName, res);
}


//设置用户选中框，主要是设置四个边界点
void GCApplication::setRectInMask()
{
	assert(!mask.empty());
	//默认全部都是背景
	mask.setTo(GC_BGD);   //GC_BGD == 0
	//考虑用户越界的特殊情况
	rect.x = max(0, rect.x);
	rect.y = max(0, rect.y);
	rect.width = min(rect.width, image->cols - rect.x);
	rect.height = min(rect.height, image->rows - rect.y);
	//将框内设置为可能的前景
	(mask(rect)).setTo(GC_PR_FGD);    //GC_PR_FGD == 3 
}


void GCApplication::setLblsInMask(int flags, Point p, bool isPr)
{
	vector<Point> *bpxls, *fpxls;
	uchar bvalue, fvalue;
	if (!isPr) //确定的前景或背景
	{
		//获取对应的容器首地址
		bpxls = &bgdPxls;
		fpxls = &fgdPxls;
		bvalue = GC_BGD;    //0
		fvalue = GC_FGD;    //1
	}
	else    //不确定的前景或背景
	{
		bpxls = &prBgdPxls;
		fpxls = &prFgdPxls;
		bvalue = GC_PR_BGD; //2
		fvalue = GC_PR_FGD; //3
	}
	//往对应的容器中推入像素
	if (flags & BGD_KEY)
	{
		bpxls->push_back(p);
		circle(mask, p, radius, bvalue, thickness);   //Set point value = 2
	}
	if (flags & FGD_KEY)
	{
		fpxls->push_back(p);
		circle(mask, p, radius, fvalue, thickness);   //Set point value = 3
	}
}


//鼠标点击事件，设置一些标志位
void GCApplication::mouseClick(int event, int x, int y, int flags, void*)
{
	switch (event)
	{
		//左击
	case CV_EVENT_LBUTTONDOWN: // Set rect or GC_BGD(GC_FGD) labels
	{
		//判断ctrl或者shift有没有被按下，分别记录在isb与isf中
		bool isb = (flags & BGD_KEY) != 0,
			isf = (flags & FGD_KEY) != 0;
		if (rectState == NOT_SET && !isb && !isf)//只按下了左键同时又没画完框
		{
			//正在画框
			rectState = IN_PROCESS;
			rect = Rect(x, y, 1, 1);
		}
		if ((isb || isf) && rectState == SET) //Set the BGD/FGD(labels).after press the "ALT" key or "SHIFT" key,and have finish drawing the rectangle
			lblsState = IN_PROCESS;
	}
	break;
	//右击
	case CV_EVENT_RBUTTONDOWN: // Set GC_PR_BGD(GC_PR_FGD) labels
	{
		bool isb = (flags & BGD_KEY) != 0,
			isf = (flags & FGD_KEY) != 0;
		if ((isb || isf) && rectState == SET) //Set the probably FGD/BGD labels
			prLblsState = IN_PROCESS;
	}
	break;
	//左键释放
	case CV_EVENT_LBUTTONUP:
		if (rectState == IN_PROCESS)
		{
			//得到一个完整的框
			rect = Rect(Point(rect.x, rect.y), Point(x, y));   //After draw the rectangle
			rectState = SET;
			setRectInMask();//设置框内框外像素的属性值
			assert(bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty());
			showImage();
		}
		if (lblsState == IN_PROCESS)
		{
			setLblsInMask(flags, Point(x, y), false);    // Draw the FGD points
			lblsState = SET;
			showImage();
		}
		break;
		//右键释放
	case CV_EVENT_RBUTTONUP:
		if (prLblsState == IN_PROCESS)
		{
			setLblsInMask(flags, Point(x, y), true); //Draw the BGD points
			prLblsState = SET;
			showImage();
		}
		break;
		//鼠标移动
	case CV_EVENT_MOUSEMOVE:
		if (rectState == IN_PROCESS)
		{
			rect = Rect(Point(rect.x, rect.y), Point(x, y));
			assert(bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty());
			showImage();   //Continue showing image
		}
		else if (lblsState == IN_PROCESS)
		{
			setLblsInMask(flags, Point(x, y), false);
			showImage();
		}
		else if (prLblsState == IN_PROCESS)
		{
			setLblsInMask(flags, Point(x, y), true);
			showImage();
		}
		break;
	}
}

//用户输入n后，执行Grab Cut运算，并返回预测的轮数
int GCApplication::nextIter()
{
	cout << iterCount << endl;
	if (isInitialized)//之前已经完成初始化，直接运算
		gc.GrabCut(*image, mask, rect, bgdModel, fgdModel, GC_CUT);
	else//还没有初始化
	{
		//用户必须先划定框框
		if (rectState != SET)
			return iterCount;
		//初次调用GrabCut，需要使用kmeans对GMM分量进行聚类
		gc.GrabCut(*image, mask, rect, bgdModel, fgdModel, GC_WITH_RECT);
		isInitialized = true;
	}
	iterCount++;

	bgdPxls.clear(); fgdPxls.clear();
	prBgdPxls.clear(); prFgdPxls.clear();

	return iterCount;
}