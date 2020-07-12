#include <iostream>
#include <opencv2/opencv.hpp>
#include "GCApplication.h"
#include <time.h>
//输出帮助信息
static void help()
{
	std::cout << "\nThis program demonstrates GrabCut segmentation -- select an object in a region\n"
		"and then grabcut will attempt to segment it out.\n"
		"Call:\n"
		"./grabcut <image_name>\n"
		"\nSelect a rectangular area around the object you want to segment\n" <<
		"\nHot keys: \n"
		"\tESC - quit the program\n"
		"\tr - restore the original image\n"
		"\tn - next iteration\n"
		"\n"
		"\tleft mouse button - set rectangle\n"
		"\n"
		"\tCTRL+left mouse button - set GC_BGD pixels\n"
		"\tSHIFT+left mouse button - set CG_FGD pixels\n"
		"\n"
		"\tCTRL+right mouse button - set GC_PR_BGD pixels\n"
		"\tSHIFT+right mouse button - set CG_PR_FGD pixels\n" << endl;
}


GCApplication gcapp;

static void on_mouse(int event, int x, int y, int flags, void* param)
{
	gcapp.mouseClick(event, x, y, flags, param);
}


int main()
{
	time_t start, end;
	//文件名
	string filename = "messi5.jpg";
	cout << "Please input the file name:";
	cin>>filename;
	//读取文件
	Mat image = imread(filename, 1);
	if (image.empty())
	{
		cout << "Couldn't read image filename " << filename << endl;
		return 1;
	}

	//设置图片大小
	Size s;
	s.height = image.rows;
	s.width = image.cols;
	resize(image, image, s);
	cout << "Pic Size:  Height:" << s.height << "  Width:" << s.width;
	//输出帮助信息
	help();

	const string winName = "image";
	namedWindow(winName.c_str(), WINDOW_AUTOSIZE);
	cvSetMouseCallback(winName.c_str(), on_mouse, 0);

	gcapp.setImageAndWinName(image, winName);
	gcapp.showImage();

	while (1) {
		int c = cvWaitKey(0);
		switch ((char)c)
		{
			//esc，退出
		case '\x1b':
			cout << "Exiting ..." << endl;
			cvDestroyWindow(winName.c_str());
			return 0;
			//reset
		case 'r':
			cout << endl;
			gcapp.reset();
			gcapp.showImage();
			break;
			//开始运行
		case 'n':
			start = clock();
			int iterCount = gcapp.getIterCount();
			cout << "**********************Begin iterator:" << iterCount << "**********************\n";
			int newIterCount = gcapp.nextIter();
			if (newIterCount > iterCount)
			{
				gcapp.showImage();
				cout << "**********************Finish iterator:" << iterCount << "**********************\n";
				end = clock();
				cout << "time=" << end-start << endl;
			}
			else
				cout << "Rectangle must be determined!" << endl;
			break;
		}
	}
}