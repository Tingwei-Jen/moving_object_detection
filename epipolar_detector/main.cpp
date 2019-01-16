#include "epipolardetector.h"  
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main()
{

	EpipolarDetector* epipolardetector = new EpipolarDetector();
    Mat img_1, img_2;
    img_1 = imread("../images/0.png", CV_LOAD_IMAGE_COLOR);   // Read the file
    img_2 = imread("../images/1.png", CV_LOAD_IMAGE_COLOR);   // Read the file

    if(! img_1.data || ! img_2.data)
    {
        cout <<  "Could not open or find the img_1 or img_2" << std::endl ;
        return -1;
    }

    int n_cols = img_1.cols;
    int n_rows = img_1.rows;

    //step1: compute F
    Mat K,del_T,F;
    epipolardetector->readParameter(K, del_T);
    epipolardetector->computeF(K, del_T, F);
    cout<<"F="<<endl<<F<<endl;

    //step2: detect feature
	Ptr<FastFeatureDetector> detector=FastFeatureDetector::create(400);
  	std::vector<KeyPoint> keypoints1;
  	detector->detect( img_1, keypoints1 );

    //step3: use optical flow to do feature matching
  	vector<Point2f> points1;
	for (int i = 0; i < keypoints1.size(); i++)
    {
		if(keypoints1[i].pt.y>150 && keypoints1[i].pt.y<270)
			points1.push_back(keypoints1[i].pt);
	} 
	cout<<"points1.size: "<<points1.size()<<endl;

    vector<Point2f> points2;
  	vector<uchar> status;
    vector<float> err;
    calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err);

    //step4: compute epipolar line in image2
    vector<Point3f> lines2; 
    epipolardetector->computeEpipolarLine(points1, F, lines2);

    //step5: compute distance error
    vector<double> dists;
    epipolardetector->distPoint2Line(points2, lines2, dists);

    //step6: draw result
    for(int i=0; i<points1.size(); i++)
    {
        cv::circle(img_1, points1[i], 5, Scalar(255,0,0), -1);      
        cv::circle(img_2, points2[i], 5, Scalar(255,0,0), -1);      
            
        if(dists[i] > 1)
        {
            cv::circle(img_2, points2[i], 5, Scalar(0,0,255), -1);    
            double a = lines2[i].x;
            double b = lines2[i].y;
            double c = lines2[i].z;
            cv::line(img_2, Point(0,-c/b), Point(n_cols, -(c+a*n_cols)/b), Scalar(0,0,255));
        } 
        else 
        {
            cv::circle(img_2, points2[i], 5, Scalar(255,0,0), -1);    
        }
    }

    imshow( "img_1", img_1 );
    imshow( "img_2", img_2 );

    waitKey(0);
    return 0;
}