#pragma once
#ifndef EPIPOLARDETECTOR_H
#define EPIPOLARDETECTOR_H

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"

class EpipolarDetector
{
public:
	EpipolarDetector();
	void readParameter(cv::Mat &K, cv::Mat &del_T);
	void computeF(cv::Mat K, cv::Mat del_T, cv::Mat& F);
	void computeEpipolarLine(std::vector<cv::Point2f> pnts1, cv::Mat F, std::vector<cv::Point3f>& lines2);
	void distPoint2Line(std::vector<cv::Point2f> pnts, std::vector<cv::Point3f> lines, std::vector<double>& dists);
};

#endif //EPIPOLARDETECTOR_H