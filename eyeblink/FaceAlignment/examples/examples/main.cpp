/*
 *
 * This file is part of the open-source SeetaFace engine, which includes three modules:
 * SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
 *
 * This file is an example of how to use SeetaFace engine for face alignment, the
 * face alignment method described in the following paper:
 *
 *
 *   Coarse-to-Fine Auto-Encoder Networks (CFAN) for Real-Time Face Alignment, 
 *   Jie Zhang, Shiguang Shan, Meina Kan, Xilin Chen. In Proceeding of the
 *   European Conference on Computer Vision (ECCV), 2014
 *
 *
 * Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
 * Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
 *
 * The codes are mainly developed by Jie Zhang (a Ph.D supervised by Prof. Shiguang Shan)
 *
 * As an open-source face recognition engine: you can redistribute SeetaFace source codes
 * and/or modify it under the terms of the BSD 2-Clause License.
 *
 * You should have received a copy of the BSD 2-Clause License along with the software.
 * If not, see < https://opensource.org/licenses/BSD-2-Clause>.
 *
 * Contact Info: you can send an email to SeetaFace@vipl.ict.ac.cn for any problems.
 *
 * Note: the above information must be kept whenever or wherever the codes are used.
 *
 */

#include <fstream>
#include <iostream>
#include <string>

#include "cv.h"
#include "highgui.h"
#include <opencv2/ml/ml.hpp>

#include "face_detection.h"
#include "face_alignment.h"
#include "Tracker.h"
#include "config.h"
#ifdef _WIN32
std::string DATA_DIR = "../../data/";
std::string MODEL_DIR = "../../model/";
#else
std::string DATA_DIR = "./data/";
std::string MODEL_DIR = "./model/";
#endif
using namespace std;
using namespace cv;

#define initial 1
#define track 2
struct res_pro
{
	bool blink = FALSE;
	IplImage *img_grayscale;
	IplImage *res;
	IplImage *res_zuo;
	//int face_num=0;
	CvSVM svm;
	CvSVM svm_zuo;
	//seeta::FaceDetection *detector;
	//seeta::FaceAlignment *point_detector;
	float score;
	float score_zuo;
	cv::Rect bbox;
	cv::Rect bbox_zuo;
};
Tracker track_initial(cv::Rect eyefield,  Mat init_frame);
void aliverdet(IplImage *frame, int count, res_pro* reut);
void track_eye(Tracker *tracker, Tracker *tracker_zuo, IplImage *frame);
//void track_init(cv::Rect eyefield, cv::Rect eyefield_zuo, Mat init_frame, Tracker *ini_tracker);
int main(int argc, char** argv)
{
	int stage = initial;
	Mat init_frame;
	Tracker *tracker=NULL;
	Tracker *tracker_zuo=NULL;
	cv::Rect eyefield;
	cv::Rect eyefield_zuo;
	res_pro reut;
	seeta::FaceDetection detector("../../../FaceDetection/model/seeta_fd_frontal_v1.0.bin");
	seeta::FaceAlignment point_detector((MODEL_DIR + "seeta_fa_v1.1.bin").c_str());
	CvCapture   *capture;
	detector.SetMinFaceSize(40);
	detector.SetScoreThresh(2.f);
	detector.SetImagePyramidScaleFactor(0.8f);
	detector.SetWindowStep(4, 4);
	reut.svm.load("svmalive-version5.xml");
	reut.svm_zuo.load("svmalive-version4.xml");
	char key='a';
	// Initialize face detection model
	IplImage *frame;
	int count = 1;
	capture = cvCaptureFromCAM(0);
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, 600);
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 500);
	cvNamedWindow("video", CV_WINDOW_NORMAL);
	//reut.detector = &detector;
	//reut.point_detector = &point_detector;
	bool info_sig = TRUE;
	string configPath = "KCFconfig.txt";
	Config conf(configPath);
	Tracker leftTracker(conf);
	Tracker rightTracker(conf);
	double totalTime = 0;
	while (key != 's')
	{	
		frame = cvQueryFrame(capture);
		if (!frame) break;
		cvFlip(frame, frame, 1);		
		if (stage == initial)
		{
			if (info_sig == TRUE)
			{
				cout << "开始定位眼睛，请坐正" << endl;
				cout << "输入r重新定位眼睛,输入y确认定位，输入结束程序" << endl;
				info_sig = FALSE;
			}
			
			IplImage *img_grayscale = cvCreateImage(cvGetSize(frame), frame->depth, 1);
			cvCvtColor(frame, img_grayscale, CV_BGR2GRAY);
			/* Always check if frame exists */
			int pts_num = 5;
			int im_width = img_grayscale->width;
			int im_height = img_grayscale->height;
			unsigned char* data = new unsigned char[im_width * im_height];
			unsigned char* data_ptr = data;
			unsigned char* image_data_ptr = (unsigned char*)img_grayscale->imageData;
			int h = 0;
			for (h = 0; h < im_height; h++) {
				memcpy(data_ptr, image_data_ptr, im_width);
				data_ptr += im_width;
				image_data_ptr += img_grayscale->widthStep;
			}
			seeta::ImageData image_data;
			image_data.data = data;
			image_data.width = im_width;
			image_data.height = im_height;
			image_data.num_channels = 1;
			seeta::FacialLandmark points[5];
			// Detect faces

			std::vector<seeta::FaceInfo> faces = detector.Detect(image_data);

			int32_t face_num = static_cast<int32_t>(faces.size());
			if (face_num > 0)
			{
				point_detector.PointDetectLandmarks(image_data, faces[0], points);
				int width = (points[1].x - points[0].x) / 3;
				int height = width / 2;
				eyefield = cv::Rect(points[1].x - width / 2, points[1].y - height / 2, width, height);
				cvRectangle(img_grayscale, cvPoint(points[1].x - width / 2, points[1].y - height / 2), cvPoint(points[1].x + width / 2, points[1].y + height / 2), Scalar(0, 0, 255), 1, 1, 0);
				eyefield_zuo = cv::Rect(points[0].x - width / 2, points[0].y - height / 2, width, height);
				cvRectangle(img_grayscale, cvPoint(points[0].x - width / 2, points[0].y - height / 2), cvPoint(points[0].x + width / 2, points[1].y + height / 2), Scalar(0, 0, 255), 1, 1, 0);
				cvShowImage("video", img_grayscale);
				key = waitKey(10);
				if (key == 'r')
					stage = initial;
				else if (key == 'y')
				{
					stage = track;
					count = 1;
					//cvSaveImage("E:\\vsfile\\jinglun\\KCF\\dataset\\sequence\\203\\1.jpg", frame);
					cout << "初始化眼睛追踪器"<<endl;
					
					//waitKey(100);
					//track_init(eyefield, eyefield_zuo, init_frame, ini_tracker);
					init_frame = Mat(frame);
					rightTracker = track_initial(eyefield,  init_frame);
					tracker = &rightTracker;
					leftTracker = track_initial(eyefield_zuo, init_frame);
					tracker_zuo = &leftTracker;
					
					cout << "初始化完成，开始追踪,请开始眨眼" << endl;
					continue;
				}	
			}
		}
		else if (stage = track)
		{
			//int tolarate=0;
			double t0 = (double)cvGetTickCount();
			track_eye(tracker, tracker_zuo, frame);		
			reut.score = tracker->CFtracker.getCFScore();
			reut.bbox.x = tracker->GetBB().XMin();
			reut.bbox.y = tracker->GetBB().YMin();
			reut.bbox.width = tracker->GetBB().Width();
			reut.bbox.height = tracker->GetBB().Height();
			reut.score_zuo = tracker_zuo->CFtracker.getCFScore();
			reut.bbox_zuo.x = tracker_zuo->GetBB().XMin();
			reut.bbox_zuo.y = tracker_zuo->GetBB().YMin();
			reut.bbox_zuo.width = tracker_zuo->GetBB().Width();
			reut.bbox_zuo.height = tracker_zuo->GetBB().Height();
			t0 = (double)cvGetTickCount() - t0;
			totalTime += t0 / (cvGetTickFrequency() * 1000);
			cout << totalTime<<endl;
			//aliverdet(frame, count, &reut);
			aliverdet(frame, count, &reut);
			cvShowImage("video", reut.img_grayscale);
			key = waitKey(1);
			count++;
			if (reut.blink == TRUE)
			{
				if (reut.score <= 0.3&&reut.score_zuo <= 0.3)
				{
					cout << "活体检测通过,完成检测，按y可再次检测" << endl;
				}
				else
				{
					stage = initial;
					count = 1;		
				}
			}
			else if (reut.score <= 0.3 || reut.score_zuo <= 0.3)
			{
				stage = initial;
				count = 1;
			}
			//	cout << "reut.score:" << reut.score << "reut.score_zuo:" << reut.score_zuo << endl;
			/*if (reut.score <= 0.3&&reut.score_zuo <= 0.3)
			{
				
			}
			else
			{
				tolarate++;
				if (tolarate>=10)
				{ 
					stage = initial;
					count = 1;
				}
				
			}*/
		}
		//if (reut.face_num > 0)
		//	count++;
		//else
		//	count = 1;
		//if (reut.blink == FALSE)
			//cout << "活体验证失败" << endl;
		//cvShowImage("video", reut.img_grayscale);
		//key=waitKey(1);
		//cvSaveImage("face.jpg", reut.img_grayscale);
		/*cvSaveImage("eye_zuo.jpg", reut.res_zuo);
		cvSaveImage("eye.jpg", reut.res);*/
	}

}
