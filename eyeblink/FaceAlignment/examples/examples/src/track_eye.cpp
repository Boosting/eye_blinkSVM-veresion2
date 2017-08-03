#include "Tracker.h"
#include "cf.h"
#include "Config.h"
#include "rect.h"
#include "drawRect.h"
#include "VOT.hpp"
#include "math.h"

#include <windows.h>
#include <iostream>
#include <fstream>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include "core.h"
using namespace std;
using namespace cv;

static Mat  rgbImg;


//返回原始RGB图像
Mat GKCgetInputRGBImage(void)
{
	return rgbImg;
}
//在图像上画结果矩形框
void rectangle(Mat& rMat, const FloatRect& rRect, const Scalar& rColour)
{
	IntRect r(rRect);
	rectangle(rMat, Point(r.XMin(), r.YMin()), Point(r.XMax(), r.YMax()), rColour, 4);
}
//在图像上画跟总结果的十字中心
void GKCrectangle(Mat& rMat, const FloatRect& rRect, const Scalar& rColour)
{
	IntRect r(rRect);
	//rectangle(rMat, Point(r.XMin(), r.YMin()), Point(r.XMax(), r.YMax()), rColour);

	vector<Point2f> corners1;

	float cen_x = r.XCentre();
	float cen_y = r.YCentre();
	//画十字星
	corners1.push_back(Point2f(cen_x - 18, cen_y));
	corners1.push_back(Point2f(cen_x, cen_y - 18));
	corners1.push_back(Point2f(cen_x + 18, cen_y));
	corners1.push_back(Point2f(cen_x, cen_y + 18));

	line(rMat, corners1[0], corners1[2], rColour, 2);
	line(rMat, corners1[1], corners1[3], rColour, 2);
}

//主函数
void track_eye(Tracker *tracker, Tracker *tracker_zuo, IplImage *frame)
{
	Mat tmp(frame);
	float scaleW = 1.f;
	float scaleH = 1.f;
	Mat result(tmp.rows, tmp.cols, CV_8UC3);
	Mat frametoshow, frame1, framevideo;
	Mat frame_tmp;
	frametoshow = tmp.clone();
	frame1 = tmp.clone();
	resize(tmp, rgbImg, Size(tmp.cols * scaleW, tmp.rows * scaleH));
	//转换为灰度图
	if (tmp.channels() == 3)
	{
		cv::cvtColor(tmp, frame1, CV_RGB2GRAY);
		resize(frame1, frame_tmp, Size(tmp.cols * scaleW, tmp.rows * scaleH));
		cv::cvtColor(frame_tmp, result, CV_GRAY2RGB);
	}
	else
	{
		frame1 = tmp.clone();
		resize(frame1, frame_tmp, Size(tmp.cols * scaleW, tmp.rows * scaleH));
		cv::cvtColor(frame_tmp, result, CV_GRAY2RGB);
	}
	//if (tracker->IsInitialised())
	//{
		//跟踪主体函数
		//double t0 = (double)cvGetTickCount();
		tracker->Track(frame_tmp);
		//double score = tracker->CFtracker.getCFScore();
		//cout << score << endl;
		//float txmin = tracker->GetBB().XMin();
		//float tymin = tracker->GetBB().YMin();
		//float twidth = tracker->GetBB().Width();
		//float theight = tracker->GetBB().Height();
		//FloatRect lastBB(txmin / scaleW, tymin / scaleH, twidth / scaleW, theight / scaleH);
		//cv::Rect grect;
		//GKCrectangle(result, tracker->GetBB(), CV_RGB(0, 255, 0));
		//GKCrectangle(frametoshow, lastBB, CV_RGB(255, 0, 0));
		tracker_zuo->Track(frame_tmp);
		//score = tracker_zuo->CFtracker.getCFScore();
		//cout << score << endl;
		//txmin = tracker_zuo->GetBB().XMin();
		//tymin = tracker_zuo->GetBB().YMin();
		//twidth = tracker_zuo->GetBB().Width();
		//theight = tracker_zuo->GetBB().Height();
		//FloatRect lastBB_zuo(txmin / scaleW, tymin / scaleH, twidth / scaleW, theight / scaleH);
		//cv::Rect grect;
		//GKCrectangle(result, tracker_zuo->GetBB(), CV_RGB(0, 255, 0));
		//GKCrectangle(frametoshow, lastBB_zuo, CV_RGB(255, 0, 0));
		//rectangle(result, tracker.GetBB(), CV_RGB(0, 255, 0));
		//rectangle(frametoshow, lastBB, CV_RGB(255, 0, 0));
	//}
	//显示预测结果图像
	//imshow("result", result);
	//waitKey(1);
	//imshow("orig", frametoshow);
	//waitKey(1);
	//return track_res;
}
