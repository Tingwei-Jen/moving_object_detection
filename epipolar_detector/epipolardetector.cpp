#include "epipolardetector.h"
#include <Eigen/Dense>
#include <iostream>
#include <math.h>

using namespace std;
using namespace cv;

EpipolarDetector::EpipolarDetector()
{
	cout<<"Construct EpipolarDetector!!"<<endl;
}

void EpipolarDetector::readParameter(cv::Mat &K, cv::Mat &del_T)
{

	K = ( Mat_<double> ( 3,3 ) << 9.597910e+02, 0, 6.960217e+02, 0, 9.569251e+02, 2.241806e+02, 0, 0, 1 );

    Mat T_w_b0 = ( Mat_<double> ( 4,4 ) <<
                0.8893391,      0.4569116,     -0.0175410,     0,
                -0.4572405,     0.8884476,     -0.0398989,     0,
                -0.0026460,     0.0435041,     0.9990497,      0,
                0,              0,             0,              1);

    Mat T_w_b1 = ( Mat_<double> ( 4,4 ) <<
                0.8896320,      0.4563254,     -0.0179469,     0.485498905415,  
                -0.4566703,     0.8886994,     -0.0408116,     -0.257592960726,
                -0.0026740,     0.0445031,     0.9990057,      0.01055908203,
                0,              0,             0,              1);

    Mat T_b_i =  ( Mat_<double> ( 4,4 ) <<
                1,              0,             0,              -1.405,  
                0,              1,             0,              0.32,
                0,              0,             1,              0.93,
                0,              0,             0,              1);

    Mat T_i_c =  ( Mat_<double> ( 4,4 ) <<
                0.0009981,      0.0084161,    0.9999641,        1.08324,  
                -0.9999904,     -0.0042517,   0.0010339,        -0.24772,
                0.0042602,      -0.9999555,   0.0084117,        0.729655,
                0,              0,            0,                1);    

    Mat T_b_c = T_b_i*T_i_c;
    Mat T_w_c0 =  T_w_b0*T_b_c;
    Mat T_w_c1 =  T_w_b1*T_b_c;
	Mat T_c0_c1 = T_w_c0.inv()*T_w_c1;

    del_T = T_c0_c1;
}

void EpipolarDetector::computeF(Mat K, Mat del_T, Mat& F)
{

    Mat t_x = ( Mat_<double> ( 3,3 ) <<
                0,                            -del_T.at<double> ( 2,3 ),   del_T.at<double> ( 1,3 ),
                del_T.at<double> ( 2,3 ),     0,                           -del_T.at<double> ( 0,3 ),
                -del_T.at<double> ( 1,3 ),    del_T.at<double> ( 0,3 ),    0);

    Mat R = ( Mat_<double> ( 3,3 ) <<
                  del_T.at<double> ( 0,0 ),   del_T.at<double> ( 0,1 ),   del_T.at<double> ( 0,2 ),
                  del_T.at<double> ( 1,0 ),   del_T.at<double> ( 1,1 ),   del_T.at<double> ( 1,2 ),
                  del_T.at<double> ( 2,0 ),   del_T.at<double> ( 2,1 ),   del_T.at<double> ( 2,2 ));

    Mat E = t_x*R;
	F = K.inv().t()*E*K.inv();

   	//clean up F
 	Eigen::MatrixXf f(3,3);
 	f(0,0) = F.at<double>(0,0); f(0,1) = F.at<double>(0,1); f(0,2) = F.at<double>(0,2);
 	f(1,0) = F.at<double>(1,0); f(1,1) = F.at<double>(1,1); f(1,2) = F.at<double>(1,2);
 	f(2,0) = F.at<double>(2,0); f(2,1) = F.at<double>(2,1); f(2,2) = F.at<double>(2,2);

	Eigen::JacobiSVD<Eigen::MatrixXf> svd(f, Eigen::ComputeFullV | Eigen::ComputeFullU); // ComputeThinU | ComputeThinV
	Eigen::MatrixXf singular_values = svd.singularValues();
	Eigen::MatrixXf left_singular_vectors = svd.matrixU();
	Eigen::MatrixXf right_singular_vectors = svd.matrixV();
    
	Eigen::MatrixXf d(3,3);
	d(0,0) = singular_values(0); d(0,1) = 0;                  d(0,2) = 0;
	d(1,0) = 0;                  d(1,1) = singular_values(1); d(1,2) = 0;
	d(2,0) = 0;                  d(2,1) = 0;                  d(2,2) = 0;

	f = left_singular_vectors*d*right_singular_vectors.transpose();
	F = ( Mat_<double> ( 3,3 ) << f(0,0), f(0,1), f(0,2), f(1,0), f(1,1), f(1,2), f(2,0), f(2,1), f(2,2));
}


void EpipolarDetector::computeEpipolarLine(std::vector<cv::Point2f> pnts1, cv::Mat F, std::vector<cv::Point3f>& lines2)
{
	lines2.clear();

	for(int i=0; i<pnts1.size(); i++)
	{
		Mat pnt1_ = ( Mat_<double> ( 3,1 ) <<  pnts1[i].x, pnts1[i].y, 1 );	
		Mat line2_ = F * pnt1_;

		Point3f line2;
		line2.x = line2_.at<double>(0,0);
		line2.y = line2_.at<double>(1,0);
		line2.z = line2_.at<double>(2,0);
		lines2.push_back(line2);
	}
}

void EpipolarDetector::distPoint2Line(std::vector<cv::Point2f> pnts, std::vector<cv::Point3f> lines, std::vector<double>& dists)
{
	dists.clear();

	for(int i=0; i<pnts.size(); i++)
	{
		double dist;
		dist = abs(pnts[i].x*lines[i].x + pnts[i].y*lines[i].y + lines[i].z)/sqrt(lines[i].x*lines[i].x+lines[i].y*lines[i].y);
		dists.push_back(dist);
	}
}