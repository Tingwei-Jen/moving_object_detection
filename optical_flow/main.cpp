#include <iostream>

#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int main()
{
	  // -- Step 1: read image
    Mat image1, image2;
    image1 = imread("../images/save_104.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file
    image2 = imread("../images/save_105.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file

    if(! image1.data || ! image2.data)
    {
        cout <<  "Could not open or find the image1 or image2" << std::endl ;
        return -1;
    }

  	//-- Step 2: Detect the keypoints using SURF/ORB/FAST Detector
  	//Ptr<SURF> detector = SURF::create( 1200 );
  	//Ptr<FeatureDetector> detector = ORB::create();
    Ptr<FastFeatureDetector> detector=FastFeatureDetector::create(300);
  	std::vector<KeyPoint> keypoints1;
  	detector->detect( image1, keypoints1 );
  	
    // --Step3: feature matching by optical flow
  	vector<Point2f> points1;
	  for (int i = 0; i < keypoints1.size(); i++)
    {
        if(keypoints1[i].pt.y<570)
          points1.push_back(keypoints1[i].pt);
    } 

    /* 
    for(int i=20; i<image1.cols-20; i=i+20)
    {
        for(int j=20; j<570; j=j+20)
            points1.push_back(Point2f(i,j));
    }
    */
    vector<Point2f> points2;
  	vector<uchar> status;
    vector<float> err;
  	calcOpticalFlowPyrLK(image1, image2, points1, points2, status, err);
	
  	for(int i=0; i<points2.size(); i++)
  	{
        cv::circle(image1, points1[i], 2, Scalar(0,0,255), -1);
        cv::circle(image2, points2[i], 2, Scalar(255,0,0), -1);
  	    arrowedLine(image2, Point(points1[i].x, points1[i].y), Point(points2[i].x, points2[i].y), Scalar(0,255,255), 1);
  	}
 	
  	// //-- Step 4: Show detected (drawn) keypoints
  	imshow("image1", image1 );
    imshow("image2", image2 );

    waitKey(0);        
	  return 0;
}