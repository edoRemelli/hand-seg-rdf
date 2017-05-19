#pragma once
class jointBilteralUpsampling
{
private:
	// ratio between target and source cols
	int dw;
	// ratio between target and source rows
	int dh;

	// look up tables
	double* lut;
	double* lut2;
	

public:
	jointBilteralUpsampling(int dw, int dh,double kernel_param);
	~jointBilteralUpsampling();
	void upsample(cv::Mat& src, cv::Mat& joint, cv::Mat& dest);
};

