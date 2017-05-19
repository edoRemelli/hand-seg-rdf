#include "jointBilteralUpsampling.h"
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>



jointBilteralUpsampling::jointBilteralUpsampling(int t_dw, int t_dh, double kernel_param)
{
	// initialize scaling operations
	dw = t_dw;
	dh = t_dh;

	// store in a table possible outcomes so to speed up upsampling
	double t_lut[256 * 3];
	double gauss_c_coeff = -0.5 / (kernel_param*kernel_param);
	for (int i = 0; i < 256 * 3; i++)
	{
		t_lut[i] = (double)std::exp(i*i*gauss_c_coeff);
	}
	// why is he doing this??
	std::vector<double> lut_(dw*dh);
	double* t_lut2 = &lut_[0];
	double gauss_s_coeff = -0.5 / (kernel_param*kernel_param);
	for (int i = 0; i < dw*dh; i++)
	{
		t_lut2[i] = (double)std::exp(i*i*gauss_s_coeff);
	}

	// copy
	lut = t_lut;
	lut2 = t_lut2;

}


jointBilteralUpsampling::~jointBilteralUpsampling()
{
}

void jointBilteralUpsampling::upsample(cv::Mat& src, cv::Mat& joint, cv::Mat& dest)
{
	if (dest.empty())dest.create(joint.size(), src.type());
	// allocate empty containers
	cv::Mat sim, jim, eim;
	// copy replicating border-top,bottom,left,right
	cv::copyMakeBorder(src, sim, 0, 1, 0, 1, cv::BORDER_REPLICATE);
	cv::copyMakeBorder(joint, jim, 0, 1, 0, 1, cv::BORDER_REPLICATE);

	// loop over rows
	for (int j = 0; j < src.rows; j++)
	{
		int n = j*dh;

		float* s = sim.ptr<float>(j);
		uchar* jnt_ = jim.ptr(n);

		// loop over cols 
		for (int i = 0, m = 0; i < src.cols; i++, m += dw)
		{
			// extract left top, right top, left below, right below on high resolution depth map

			// left top
			const uchar ltd = jnt_[m];

			// right top
			const uchar rtd = jnt_[(m + dw)];

			// left below
			const uchar lbd = jnt_[(m + jim.cols*dh)];

			// right below
			const uchar rbd = jnt_[(m + dw + jim.cols*dh)];

			// extract left top, right top, left below, right below on low resolution probability map
			const float ltp = s[i];
			const float rtp = s[i + 1];
			const float lbp = s[i + sim.cols];
			const float rbp = s[i + 1 + sim.cols];

			// fill up neighborhood 
			for (int l = 0; l < dh; l++)
			{
				// pointer to destination high resolution probability map
				float* d = dest.ptr<float>(n + l);
				// pointer to destination high resolution depth map
				uchar* jnt = jim.ptr(n + l);

				for (int k = 0; k < dw; k++)
				{
					// depth of pixel on high resolution depth map
					const uchar dep = jnt[(m + k)];
					// compute weights
					double wlt = lut2[k + l] * lut[abs(ltd - dep)];
					double wrt = lut2[dw - k + l] * lut[abs(rtd - dep)];
					double wlb = lut2[k + dh - l] * lut[abs(lbd - dep)];
					double wrb = lut2[dw - k + dh - l] * lut[abs(rbd - dep)];
					// weighted interpolation
					d[m + k] = (wlt*ltp + wrt*rtp + wlb*lbp + wrb*rbp) / (wlt + wrt + wlb + wrb);
				}
			}
		}
	}
}
