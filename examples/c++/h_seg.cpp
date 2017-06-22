#include <fertilized/fertilized.h>
#include "fertilized/ndarray.h"
#include "fertilized/global.h"


#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <Eigen/Core>

#include <memory>
#include <vector>
#include <functional>
#include <string>
#include <fstream>
#include <algorithm>


# define SIZE_DATASET 29819
//# define SIZE_DATASET 12366
# define SIZE_TEST 181
// feature extraction params
# define N_FEAT 8.0
// how many points? n_features = (2*N_FEAT +1)*(2*N_FEAT +1)
# define N_FEAT_PER_NODE 100.0
// how far away should we shoot when computing features - should be proportional to N_FEAT 
# define DELTA 12000.0

// 170
# define N_SAMPLES_0 160
# define N_SAMPLES_1 160
# define N_SAMPLES_2 160
// resolution on which process frame
# define SRC_COLS 80
# define SRC_ROWS 60
// htrack's resolution
# define OSRC_COLS 320
# define OSRC_ROWS 240
// depth of 0-depth pixels
# define BACKGROUND_DEPTH 3000.0
// rdf parameters 
# define THRESHOLD 0.7
# define THRESHOLD_WRIST 0.6
# define MAX_DEPTH 20
# define N_TREES 5
# define N_THREADS 16
// post processing parameters
# define DILATION_SIZE 9
# define DEPTH_PP 550
# define KERNEL_SIZE 3
# define GET_CLOSER_TO_SENSOR 600


using namespace fertilized;
using namespace std;
using namespace std::chrono;


#include "pxcsensemanager.h"
#include "pxcsession.h"
#include "pxcprojection.h"

enum Mode { TRAIN, TEST, LIVE };

// sensor resolution
int D_width = 640;
int D_height = 480;
// flag for looking at TRAINING DATA
bool debug_data = false;
Mode mode = LIVE;
float crop_radius = 150.0;


PXCSenseManager * initialize() {

	PXCSenseManager *sense_manager = PXCSenseManager::CreateInstance();
	if (!sense_manager) {
		wprintf_s(L"Unable to create the PXCSenseManager\n");
		return false;
	}
	sense_manager->EnableStream(PXCCapture::STREAM_TYPE_COLOR, D_width, D_height, 60);
	sense_manager->EnableStream(PXCCapture::STREAM_TYPE_DEPTH, D_width, D_height, 60);
	sense_manager->Init();

	PXCSession *session = PXCSession::CreateInstance();
	PXCSession::ImplDesc desc, desc1;
	memset(&desc, 0, sizeof(desc));
	desc.group = PXCSession::IMPL_GROUP_SENSOR;
	desc.subgroup = PXCSession::IMPL_SUBGROUP_VIDEO_CAPTURE;
	if (session->QueryImpl(&desc, 0, &desc1) < PXC_STATUS_NO_ERROR) return false;

	PXCCapture * capture;
	pxcStatus status = session->CreateImpl<PXCCapture>(&desc1, &capture);
	if (status != PXC_STATUS_NO_ERROR) {
		exit(0);
	}

	PXCCapture::Device* device;
	device = capture->CreateDevice(0);
	return sense_manager;
}

template<class T>
void jointBilateralUpsample_(cv::Mat& src, cv::Mat& joint, cv::Mat& dest, double sigma_c, double sigma_s)
{
	if (dest.empty())dest.create(joint.size(), src.type());
	// allocate empty containers
	cv::Mat sim, jim, eim;
	// copy replicating border-top,bottom,left,right
	copyMakeBorder(src, sim, 0, 1, 0, 1, cv::BORDER_REPLICATE);
	copyMakeBorder(joint, jim, 0, 1, 0, 1, cv::BORDER_REPLICATE);
	// get required upscaling 
	const int dw = (joint.cols) / (src.cols);
	const int dh = (joint.rows) / (src.rows);

	// store in a table possible outcomes of gaussian operation - MOVE OUTSIDE TO OPTIMIZE EVEN MORE
	double lut[256 * 3];
	double gauss_c_coeff = -0.5 / (sigma_c*sigma_c);
	for (int i = 0; i < 256 * 3; i++)
	{
		lut[i] = (double)std::exp(i*i*gauss_c_coeff);
	}
	vector<double> lut_(dw*dh);
	double* lut2 = &lut_[0];		
	double gauss_s_coeff = -0.5 / (sigma_s*sigma_s);
	for (int i = 0; i < dw*dh; i++)
	{
		lut2[i] = (double)std::exp(i*i*gauss_s_coeff);
	}

	// loop over rows
	for (int j = 0; j < src.rows; j++)
	{
		int n = j*dh;

		T* s = sim.ptr<T>(j);
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
			const uchar rbd = jnt_[ (m + dw + jim.cols*dh) ];

			// extract left top, right top, left below, right below on low resolution probability map
			const T ltp = s[i];
			const T rtp = s[i + 1];
			const T lbp = s[i + sim.cols];
			const T rbp = s[i + 1 + sim.cols];

			// fill up neighborhood 
			for (int l = 0; l < dh; l++)
			{
				// pointer to destination high resolution probability map
				T* d = dest.ptr<T>(n + l);
				// pointer to destination high resolution depth map
				uchar* jnt = jim.ptr(n + l);

				for (int k = 0; k < dw; k++)
				{
					// depth of pixel on high resolution depth map

					const uchar dep = jnt[ (m + k) ];
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

Eigen::Vector3f point_at_depth_pixel(cv::Mat& depth, int x, int y, Eigen::Matrix<float, 3, 3> iproj) {
	int z = depth.at<ushort>(y, x);
	Eigen::Vector3f v = Eigen::Vector3f( x * z, (240.0- y - 1) * z, z);
	return iproj * v;
}

Eigen::Vector3f point_at_depth_pixel_for_testing(cv::Mat& depth, int x, int y, Eigen::Matrix<float, 3, 3> iproj) {
	int z = depth.at<float>(y, x);
	Eigen::Vector3f v = Eigen::Vector3f(x * z, (240.0 - y - 1) * z, z);
	return iproj * v;
}

int main() {

	
	// define camera matrix
	Eigen::Matrix<float, 3, 3> cam_matrix = Eigen::Matrix<float, 3, 3>::Zero();
	cam_matrix(0, 0) = 463.889; /// FocalLength X
	cam_matrix(1, 1) = 463.889; /// FocalLength Y
	cam_matrix(0, 2) = 320;      /// CameraCenter X
	cam_matrix(1, 2) = 240;     /// CameraCenter Y
	cam_matrix(2, 2) = 1.0;

	Eigen::Matrix<float, 3, 3> iproj = cam_matrix.inverse();

	int n_features = (2 * N_FEAT + 1)*(2 * N_FEAT + 1);

	cv::Mat src_X;
	cv::Mat src_Y;

	cv::Mat converted_src;
	std::vector<cv::Point> locations_hand;
	std::vector<cv::Point> locations_arm;
	std::vector<cv::Point> locations_wrist;
	std::vector<cv::Point> locations_background;
	std::vector<cv::Point> wrist;
	cv::Mat mask;


	auto soil = Soil<float, float, uint, Result_Types::probabilities>();

	if(mode == TRAIN)
	{
	// create fertilized arrays
	Array<float, 2, 2> X = allocate(SIZE_DATASET*(N_SAMPLES_0 + N_SAMPLES_1 + N_SAMPLES_2), n_features);
	Array<uint, 2, 2> Y = allocate(SIZE_DATASET*(N_SAMPLES_0 + N_SAMPLES_1 + N_SAMPLES_2), 1);
	std::cout << "loading data..." << std::endl;
	// read data
	cv::String train_folder = "C:/Users/remelli/Desktop/RDF/Train__";
	//cv::String train_folder = "C:/Users/remelli/Desktop/test";
	std::vector<cv::String> filenames_X;
	std::vector<cv::String> filenames_Y;
	cv::glob(train_folder + "/X", filenames_X);
	cv::glob(train_folder + "/Y", filenames_Y);
	for (size_t i = 0; i < SIZE_DATASET; i++)
	{
		cv::Mat src_X_ds = cv::Mat(cv::Size(SRC_COLS, SRC_ROWS), CV_16UC1, cv::Scalar(0));
		cv::Mat src_Y_ds = cv::Mat(cv::Size(SRC_COLS, SRC_ROWS), CV_8UC1, cv::Scalar(0));

		std::cout << "Loading frame..." << std::endl;
		std::cout << filenames_X[i] << std::endl;
		// read depth map
		src_X = cv::imread(filenames_X[i], CV_LOAD_IMAGE_UNCHANGED);
		cv::medianBlur(src_X, src_X, KERNEL_SIZE);

		src_X.convertTo(src_X, CV_32F);
		// read ground truth
		std::cout << filenames_Y[i] << std::endl;
		src_Y = cv::imread(filenames_Y[i], CV_LOAD_IMAGE_GRAYSCALE);
		// downsample 
		cv::resize(src_X, src_X_ds, cv::Size(SRC_COLS, SRC_ROWS), 0, 0, cv::INTER_NEAREST);
		cv::resize(src_Y, src_Y_ds, cv::Size(SRC_COLS, SRC_ROWS), 0, 0, cv::INTER_NEAREST);

		// prepare vector for fast element acces
		float* ptr = (float*)src_X_ds.data;
		size_t elem_step = src_X_ds.step / sizeof(float);
		// for debugging
		if (debug_data)
			cv::cvtColor(src_Y_ds, converted_src, CV_GRAY2RGB);


		// extract samples from hand region
		locations_hand.clear();
		if (cv::countNonZero(src_Y_ds > 200) > 0)
		{
		cv::findNonZero(src_Y_ds > 200, locations_hand);
		// shuffle vector 
		std::random_shuffle(locations_hand.begin(), locations_hand.end());
		auto old_count = locations_hand.size();
		locations_hand.resize(2 * old_count);
		std::copy_n(locations_hand.begin(), old_count, locations_hand.begin() + old_count);


		std::cout << locations_hand.size() << std::endl;

		// display for debugging
		for (size_t j = 0; j < std::min<int>(locations_hand.size(), N_SAMPLES_1); j++)
		{
			// draw sample
			if (debug_data)
				cv::circle(converted_src, locations_hand[j], 1, CV_RGB(255, 0, 0), 2);
			// depth of current pixel
			float d = (float)ptr[elem_step*locations_hand[j].y + locations_hand[j].x];
			//std::cout << d << std::endl;
			for (size_t k = 0; k < (2 * N_FEAT + 1); k++)
			{
				int idx_x = locations_hand[j].x + (int)(DELTA / d) * ((k - N_FEAT) / N_FEAT);
				// build feature vector
				for (size_t l = 0; l < (2 * N_FEAT + 1); l++)
				{
					int idx_y = locations_hand[j].y + (int)(DELTA / d) * ((l - N_FEAT) / N_FEAT);

					// read data
					if (idx_x < 0 || idx_x > SRC_COLS || idx_y < 0 || idx_y > SRC_ROWS)
					{
						X[i*(N_SAMPLES_0 + N_SAMPLES_1 + N_SAMPLES_2) + j][k*(2 * N_FEAT + 1) + l] = BACKGROUND_DEPTH - d;
						continue;
					}
					if (debug_data)
						cv::circle(converted_src, cv::Point(idx_x, idx_y), 1, CV_RGB(0, 255, 255), 1);
					float d_idx = (float)ptr[elem_step*idx_y + idx_x];
					if (d_idx == 0)
					{
						X[i*(N_SAMPLES_0 + N_SAMPLES_1 + N_SAMPLES_2) + j][k*(2 * N_FEAT + 1) + l] = BACKGROUND_DEPTH - d;
						continue;
					}
					X[i*(N_SAMPLES_0 + N_SAMPLES_1 + N_SAMPLES_2) + j][k*(2 * N_FEAT + 1) + l] = d_idx - d;
				}
			}
			// build ground truth vector			
			Y[i*(N_SAMPLES_0 + N_SAMPLES_1 + N_SAMPLES_2) + j][0] = 1;
		}
	}

		

		// extract samples from background region
		locations_arm.clear();
		cv::bitwise_and(src_X_ds > 10, src_Y_ds == 0, mask);
		cv::bitwise_or(src_Y_ds == 120, mask == 255, mask);
		
		cv::findNonZero(mask, locations_arm);
		
		// shuffle vector 
		std::random_shuffle(locations_arm.begin(), locations_arm.end());
		
		//old_count = locations_arm.size();
		//locations_arm.resize(2 * old_count);
		//std::copy_n(locations_arm.begin(), old_count, locations_arm.begin() + old_count);
		

		std::cout << locations_arm.size() << std::endl;

		// display for debugging
		for (size_t j = 0; j < std::min<int>(locations_arm.size(), N_SAMPLES_0); j++)
		{
			if (debug_data)
				cv::circle(converted_src, locations_arm[j], 1, CV_RGB(0, 0, 255), 2);

			// depth of current pixel
			float d = (float)ptr[elem_step*locations_arm[j].y + locations_arm[j].x];
			//std::cout << d << std::endl;
			for (size_t k = 0; k < (2 * N_FEAT + 1); k++)
			{
				int idx_x = locations_arm[j].x + (int)(DELTA / d) * ((k - N_FEAT) / N_FEAT);

				for (size_t l = 0; l < (2 * N_FEAT + 1); l++)
				{
					int idx_y = locations_arm[j].y + (int)(DELTA / d) * ((l - N_FEAT) / N_FEAT);
					// read data
					if (idx_x < 0 || idx_x > SRC_COLS || idx_y < 0 || idx_y > SRC_ROWS)
					{
						X[i*(N_SAMPLES_0 + N_SAMPLES_1 + N_SAMPLES_2) + N_SAMPLES_1 + j][k*(2 * N_FEAT + 1) + l] = BACKGROUND_DEPTH - d;
						continue;
					}
					if (debug_data)
						cv::circle(converted_src, cv::Point(idx_x, idx_y), 1, CV_RGB(255, 0, 255), 1);
					float d_idx = (float)ptr[elem_step*idx_y + idx_x];
					if (d_idx == 0)
					{
						X[i*(N_SAMPLES_0 + N_SAMPLES_1 + N_SAMPLES_2) + N_SAMPLES_1 + j][k*(2 * N_FEAT + 1) + l] = BACKGROUND_DEPTH - d;
						continue;
					}
					X[i*(N_SAMPLES_0 + N_SAMPLES_1 + N_SAMPLES_2) + N_SAMPLES_1 + j][k*(2 * N_FEAT + 1) + l] = d_idx - d;
					
				}
			}
			Y[i*(N_SAMPLES_0 + N_SAMPLES_1 + N_SAMPLES_2) + N_SAMPLES_1 + j][0] = 0;
			
		}
		

		// extract samples from wrist region
		locations_arm.clear();
		if (cv::countNonZero(src_Y_ds == 60) > 0)
		{
		cv::findNonZero(src_Y_ds == 60, locations_wrist);
		// shuffle vector 
		std::random_shuffle(locations_wrist.begin(), locations_wrist.end());

		auto old_count = locations_wrist.size();
		locations_wrist.resize(2 * old_count);
		std::copy_n(locations_wrist.begin(), old_count, locations_wrist.begin() + old_count);
		old_count = locations_wrist.size();
		locations_wrist.resize(2 * old_count);
		std::copy_n(locations_wrist.begin(), old_count, locations_wrist.begin() + old_count);

		std::cout << locations_wrist.size() << std::endl;
		// display for debugging
		for (size_t j = 0; j < std::min<int>(locations_wrist.size(), N_SAMPLES_2); j++)
		{
			if (debug_data)
				cv::circle(converted_src, locations_wrist[j], 1, CV_RGB(0, 0, 255), 2);

			// depth of current pixel
			float d = (float)ptr[elem_step*locations_wrist[j].y + locations_wrist[j].x];
			//std::cout << d << std::endl;
			for (size_t k = 0; k < (2 * N_FEAT + 1); k++)
			{
				int idx_x = locations_wrist[j].x + (int)(DELTA / d) * ((k - N_FEAT) / N_FEAT);

				for (size_t l = 0; l < (2 * N_FEAT + 1); l++)
				{
					int idx_y = locations_wrist[j].y + (int)(DELTA / d) * ((l - N_FEAT) / N_FEAT);
					// read data
					if (idx_x < 0 || idx_x > SRC_COLS || idx_y < 0 || idx_y > SRC_ROWS)
					{
						X[i*(N_SAMPLES_0 + N_SAMPLES_1 + N_SAMPLES_2) + N_SAMPLES_1 + N_SAMPLES_0 + j][k*(2 * N_FEAT + 1) + l] = BACKGROUND_DEPTH - d;
						continue;
					}
					if (debug_data)
						cv::circle(converted_src, cv::Point(idx_x, idx_y), 1, CV_RGB(255, 255, 255), 1);
					float d_idx = (float)ptr[elem_step*idx_y + idx_x];
					if (d_idx == 0)
					{
						X[i*(N_SAMPLES_0 + N_SAMPLES_1 + N_SAMPLES_2) + N_SAMPLES_1 + N_SAMPLES_0 + j][k*(2 * N_FEAT + 1) + l] = BACKGROUND_DEPTH - d;
						continue;
					}
					X[i*(N_SAMPLES_0 + N_SAMPLES_1 + N_SAMPLES_2) + N_SAMPLES_1 + N_SAMPLES_0 + j][k*(2 * N_FEAT + 1) + l] = d_idx - d;
				}
			}
			Y[i*(N_SAMPLES_0 + N_SAMPLES_1 + N_SAMPLES_2) + N_SAMPLES_1 + N_SAMPLES_0 + j][0] = 2;
		}
		}

		if (debug_data)
		{
			double min;
			double max;

			cv::imshow("figaaa", src_Y);
			cv::minMaxIdx(src_Y_ds, &min, &max);
			std::cout << min << std::endl;
			std::cout << max << std::endl;

			cv::imshow("frame", converted_src);
			cv::minMaxIdx(src_X, &min, &max);
			cv::Mat normalized_depth;
			src_X.convertTo(normalized_depth, CV_8UC1, 255.0 / (max - min), -min);
			cv::imshow("depth map", normalized_depth);

			cv::waitKey(0);
		}
	}

	/*int last_ind = SIZE_DATASET*(N_SAMPLES_0 + N_SAMPLES_1 + N_SAMPLES_2);

	// IMAGES ROTATED LEFT
	for (size_t i = 0; i < SIZE_DATASET; i++)
	{
		cv::Mat src_X_ds = cv::Mat(cv::Size(SRC_COLS, SRC_ROWS), CV_16UC1, cv::Scalar(0));
		cv::Mat src_Y_ds = cv::Mat(cv::Size(SRC_COLS, SRC_ROWS), CV_8UC1, cv::Scalar(0));

		std::cout << "Loading frame..." << std::endl;
		std::cout << filenames_X[i] << std::endl;
		// read depth map
		src_X = cv::imread(filenames_X[i], CV_LOAD_IMAGE_UNCHANGED);
		cv::medianBlur(src_X, src_X, KERNEL_SIZE);
		src_X.convertTo(src_X, CV_32F);
		// read ground truth
		std::cout << filenames_Y[i] << std::endl;
		src_Y = cv::imread(filenames_Y[i], CV_LOAD_IMAGE_GRAYSCALE);
		// rotate
		cv::Point2f pc(src_X.cols / 2., src_X.rows / 2.);
		cv::Mat r = cv::getRotationMatrix2D(pc, -10, 1.0);
		cv::warpAffine(src_X, src_X, r, src_X.size());
		cv::warpAffine(src_Y, src_Y, r, src_X.size());
		// downsample 
		cv::resize(src_X, src_X_ds, cv::Size(SRC_COLS, SRC_ROWS), 0, 0, cv::INTER_NEAREST);
		cv::resize(src_Y, src_Y_ds, cv::Size(SRC_COLS, SRC_ROWS), 0, 0, cv::INTER_NEAREST);

		// prepare vector for fast element acces
		float* ptr = (float*)src_X_ds.data;
		size_t elem_step = src_X_ds.step / sizeof(float);
		// for debugging
		if (debug_data)
			cv::cvtColor(src_Y_ds, converted_src, CV_GRAY2RGB);


		// extract samples from hand region
		locations_hand.clear();
		cv::findNonZero(src_Y_ds > 200, locations_hand);
		// shuffle vector 
		std::random_shuffle(locations_hand.begin(), locations_hand.end());
		auto old_count = locations_hand.size();
		locations_hand.resize(2 * old_count);
		std::copy_n(locations_hand.begin(), old_count, locations_hand.begin() + old_count);


		std::cout << locations_hand.size() << std::endl;

		// display for debugging
		for (size_t j = 0; j < std::min<int>(locations_hand.size(), N_SAMPLES_1); j++)
		{
			// draw sample
			if (debug_data)
				cv::circle(converted_src, locations_hand[j], 1, CV_RGB(255, 0, 0), 2);
			// depth of current pixel
			float d = (float)ptr[elem_step*locations_hand[j].y + locations_hand[j].x];
			//std::cout << d << std::endl;
			for (size_t k = 0; k < (2 * N_FEAT + 1); k++)
			{
				int idx_x = locations_hand[j].x + (int)(DELTA / d) * ((k - N_FEAT) / N_FEAT);
				// build feature vector
				for (size_t l = 0; l < (2 * N_FEAT + 1); l++)
				{
					int idx_y = locations_hand[j].y + (int)(DELTA / d) * ((l - N_FEAT) / N_FEAT);

					// read data
					if (idx_x < 0 || idx_x > SRC_COLS || idx_y < 0 || idx_y > SRC_ROWS)
					{
						X[last_ind+i*(N_SAMPLES_0 + N_SAMPLES_1 + N_SAMPLES_2) + j][k*(2 * N_FEAT + 1) + l] = BACKGROUND_DEPTH - d;
						continue;
					}
					if (debug_data)
						cv::circle(converted_src, cv::Point(idx_x, idx_y), 1, CV_RGB(0, 255, 255), 1);
					float d_idx = (float)ptr[elem_step*idx_y + idx_x];
					if (d_idx == 0)
					{
						X[last_ind+i*(N_SAMPLES_0 + N_SAMPLES_1 + N_SAMPLES_2) + j][k*(2 * N_FEAT + 1) + l] = BACKGROUND_DEPTH - d;
						continue;
					}
					X[last_ind+i*(N_SAMPLES_0 + N_SAMPLES_1 + N_SAMPLES_2) + j][k*(2 * N_FEAT + 1) + l] = d_idx - d;
				}
			}
			// build ground truth vector			
			Y[last_ind+i*(N_SAMPLES_0 + N_SAMPLES_1 + N_SAMPLES_2) + j][0] = 1;
		}



		// extract samples from background region
		locations_arm.clear();
		cv::bitwise_and(src_X_ds > 10, src_Y_ds == 0, mask);
		cv::bitwise_or(src_Y_ds == 120, mask == 255, mask);

		cv::findNonZero(mask, locations_arm);

		// shuffle vector 
		std::random_shuffle(locations_arm.begin(), locations_arm.end());

		//old_count = locations_arm.size();
		//locations_arm.resize(2 * old_count);
		//std::copy_n(locations_arm.begin(), old_count, locations_arm.begin() + old_count);


		std::cout << locations_arm.size() << std::endl;

		// display for debugging
		for (size_t j = 0; j < std::min<int>(locations_arm.size(), N_SAMPLES_0); j++)
		{
			if (debug_data)
				cv::circle(converted_src, locations_arm[j], 1, CV_RGB(0, 0, 255), 2);

			// depth of current pixel
			float d = (float)ptr[elem_step*locations_arm[j].y + locations_arm[j].x];
			//std::cout << d << std::endl;
			for (size_t k = 0; k < (2 * N_FEAT + 1); k++)
			{
				int idx_x = locations_arm[j].x + (int)(DELTA / d) * ((k - N_FEAT) / N_FEAT);

				for (size_t l = 0; l < (2 * N_FEAT + 1); l++)
				{
					int idx_y = locations_arm[j].y + (int)(DELTA / d) * ((l - N_FEAT) / N_FEAT);
					// read data
					if (idx_x < 0 || idx_x > SRC_COLS || idx_y < 0 || idx_y > SRC_ROWS)
					{
						X[last_ind+i*(N_SAMPLES_0 + N_SAMPLES_1 + N_SAMPLES_2) + N_SAMPLES_1 + j][k*(2 * N_FEAT + 1) + l] = BACKGROUND_DEPTH - d;
						continue;
					}
					if (debug_data)
						cv::circle(converted_src, cv::Point(idx_x, idx_y), 1, CV_RGB(255, 0, 255), 1);
					float d_idx = (float)ptr[elem_step*idx_y + idx_x];
					if (d_idx == 0)
					{
						X[last_ind+i*(N_SAMPLES_0 + N_SAMPLES_1 + N_SAMPLES_2) + N_SAMPLES_1 + j][k*(2 * N_FEAT + 1) + l] = BACKGROUND_DEPTH - d;
						continue;
					}
					X[last_ind+i*(N_SAMPLES_0 + N_SAMPLES_1 + N_SAMPLES_2) + N_SAMPLES_1 + j][k*(2 * N_FEAT + 1) + l] = d_idx - d;

				}
			}
			Y[last_ind+i*(N_SAMPLES_0 + N_SAMPLES_1 + N_SAMPLES_2) + N_SAMPLES_1 + j][0] = 0;

		}


		// extract samples from wrist region
		locations_arm.clear();
		cv::findNonZero(src_Y_ds == 60, locations_wrist);
		// shuffle vector 
		std::random_shuffle(locations_wrist.begin(), locations_wrist.end());

		old_count = locations_wrist.size();
		locations_wrist.resize(2 * old_count);
		std::copy_n(locations_wrist.begin(), old_count, locations_wrist.begin() + old_count);
		old_count = locations_wrist.size();
		locations_wrist.resize(2 * old_count);
		std::copy_n(locations_wrist.begin(), old_count, locations_wrist.begin() + old_count);

		std::cout << locations_wrist.size() << std::endl;

		// display for debugging
		for (size_t j = 0; j < std::min<int>(locations_wrist.size(), N_SAMPLES_2); j++)
		{
			if (debug_data)
				cv::circle(converted_src, locations_wrist[j], 1, CV_RGB(0, 0, 255), 2);

			// depth of current pixel
			float d = (float)ptr[elem_step*locations_wrist[j].y + locations_wrist[j].x];
			//std::cout << d << std::endl;
			for (size_t k = 0; k < (2 * N_FEAT + 1); k++)
			{
				int idx_x = locations_wrist[j].x + (int)(DELTA / d) * ((k - N_FEAT) / N_FEAT);

				for (size_t l = 0; l < (2 * N_FEAT + 1); l++)
				{
					int idx_y = locations_wrist[j].y + (int)(DELTA / d) * ((l - N_FEAT) / N_FEAT);
					// read data
					if (idx_x < 0 || idx_x > SRC_COLS || idx_y < 0 || idx_y > SRC_ROWS)
					{
						X[last_ind+i*(N_SAMPLES_0 + N_SAMPLES_1 + N_SAMPLES_2) + N_SAMPLES_1 + N_SAMPLES_0 + j][k*(2 * N_FEAT + 1) + l] = BACKGROUND_DEPTH - d;
						continue;
					}
					if (debug_data)
						cv::circle(converted_src, cv::Point(idx_x, idx_y), 1, CV_RGB(255, 255, 255), 1);
					float d_idx = (float)ptr[elem_step*idx_y + idx_x];
					if (d_idx == 0)
					{
						X[last_ind + i*(N_SAMPLES_0 + N_SAMPLES_1 + N_SAMPLES_2) + N_SAMPLES_1 + N_SAMPLES_0 + j][k*(2 * N_FEAT + 1) + l] = BACKGROUND_DEPTH - d;
						continue;
					}
					X[last_ind + i*(N_SAMPLES_0 + N_SAMPLES_1 + N_SAMPLES_2) + N_SAMPLES_1 + N_SAMPLES_0 + j][k*(2 * N_FEAT + 1) + l] = d_idx - d;
				}
			}
			Y[last_ind + i*(N_SAMPLES_0 + N_SAMPLES_1 + N_SAMPLES_2) + N_SAMPLES_1 + N_SAMPLES_0 + j][0] = 2;
		}

		if (debug_data)
		{
			double min;
			double max;

			cv::imshow("figaaa", src_Y);
			cv::minMaxIdx(src_Y_ds, &min, &max);
			std::cout << min << std::endl;
			std::cout << max << std::endl;

			cv::imshow("frame", converted_src);
			cv::minMaxIdx(src_X, &min, &max);
			cv::Mat normalized_depth;
			src_X.convertTo(normalized_depth, CV_8UC1, 255.0 / (max - min), -min);
			cv::imshow("depth map", normalized_depth);

			cv::waitKey(0);
		}
	}


	last_ind = 2*SIZE_DATASET*(N_SAMPLES_0 + N_SAMPLES_1 + N_SAMPLES_2);

	// IMAGES ROTATED LEFT
	for (size_t i = 0; i < SIZE_DATASET; i++)
	{
		cv::Mat src_X_ds = cv::Mat(cv::Size(SRC_COLS, SRC_ROWS), CV_16UC1, cv::Scalar(0));
		cv::Mat src_Y_ds = cv::Mat(cv::Size(SRC_COLS, SRC_ROWS), CV_8UC1, cv::Scalar(0));

		std::cout << "Loading frame..." << std::endl;
		std::cout << filenames_X[i] << std::endl;
		// read depth map
		src_X = cv::imread(filenames_X[i], CV_LOAD_IMAGE_UNCHANGED);
		cv::medianBlur(src_X, src_X, KERNEL_SIZE);
		src_X.convertTo(src_X, CV_32F);
		// read ground truth
		std::cout << filenames_Y[i] << std::endl;
		src_Y = cv::imread(filenames_Y[i], CV_LOAD_IMAGE_GRAYSCALE);
		// rotate
		cv::Point2f pc(src_X.cols / 2., src_X.rows / 2.);
		cv::Mat r = cv::getRotationMatrix2D(pc, +10, 1.0);
		cv::warpAffine(src_X, src_X, r, src_X.size());
		cv::warpAffine(src_Y, src_Y, r, src_X.size());
		// downsample 
		cv::resize(src_X, src_X_ds, cv::Size(SRC_COLS, SRC_ROWS), 0, 0, cv::INTER_NEAREST);
		cv::resize(src_Y, src_Y_ds, cv::Size(SRC_COLS, SRC_ROWS), 0, 0, cv::INTER_NEAREST);

		// prepare vector for fast element acces
		float* ptr = (float*)src_X_ds.data;
		size_t elem_step = src_X_ds.step / sizeof(float);
		// for debugging
		if (debug_data)
			cv::cvtColor(src_Y_ds, converted_src, CV_GRAY2RGB);


		// extract samples from hand region
		locations_hand.clear();
		cv::findNonZero(src_Y_ds > 200, locations_hand);
		// shuffle vector 
		std::random_shuffle(locations_hand.begin(), locations_hand.end());
		auto old_count = locations_hand.size();
		locations_hand.resize(2 * old_count);
		std::copy_n(locations_hand.begin(), old_count, locations_hand.begin() + old_count);


		std::cout << locations_hand.size() << std::endl;

		// display for debugging
		for (size_t j = 0; j < std::min<int>(locations_hand.size(), N_SAMPLES_1); j++)
		{
			// draw sample
			if (debug_data)
				cv::circle(converted_src, locations_hand[j], 1, CV_RGB(255, 0, 0), 2);
			// depth of current pixel
			float d = (float)ptr[elem_step*locations_hand[j].y + locations_hand[j].x];
			//std::cout << d << std::endl;
			for (size_t k = 0; k < (2 * N_FEAT + 1); k++)
			{
				int idx_x = locations_hand[j].x + (int)(DELTA / d) * ((k - N_FEAT) / N_FEAT);
				// build feature vector
				for (size_t l = 0; l < (2 * N_FEAT + 1); l++)
				{
					int idx_y = locations_hand[j].y + (int)(DELTA / d) * ((l - N_FEAT) / N_FEAT);

					// read data
					if (idx_x < 0 || idx_x > SRC_COLS || idx_y < 0 || idx_y > SRC_ROWS)
					{
						X[last_ind + i*(N_SAMPLES_0 + N_SAMPLES_1 + N_SAMPLES_2) + j][k*(2 * N_FEAT + 1) + l] = BACKGROUND_DEPTH - d;
						continue;
					}
					if (debug_data)
						cv::circle(converted_src, cv::Point(idx_x, idx_y), 1, CV_RGB(0, 255, 255), 1);
					float d_idx = (float)ptr[elem_step*idx_y + idx_x];
					if (d_idx == 0)
					{
						X[last_ind + i*(N_SAMPLES_0 + N_SAMPLES_1 + N_SAMPLES_2) + j][k*(2 * N_FEAT + 1) + l] = BACKGROUND_DEPTH - d;
						continue;
					}
					X[last_ind + i*(N_SAMPLES_0 + N_SAMPLES_1 + N_SAMPLES_2) + j][k*(2 * N_FEAT + 1) + l] = d_idx - d;
				}
			}
			// build ground truth vector			
			Y[last_ind + i*(N_SAMPLES_0 + N_SAMPLES_1 + N_SAMPLES_2) + j][0] = 1;
		}



		// extract samples from background region
		locations_arm.clear();
		cv::bitwise_and(src_X_ds > 10, src_Y_ds == 0, mask);
		cv::bitwise_or(src_Y_ds == 120, mask == 255, mask);

		cv::findNonZero(mask, locations_arm);

		// shuffle vector 
		std::random_shuffle(locations_arm.begin(), locations_arm.end());

		//old_count = locations_arm.size();
		//locations_arm.resize(2 * old_count);
		//std::copy_n(locations_arm.begin(), old_count, locations_arm.begin() + old_count);


		std::cout << locations_arm.size() << std::endl;

		// display for debugging
		for (size_t j = 0; j < std::min<int>(locations_arm.size(), N_SAMPLES_0); j++)
		{
			if (debug_data)
				cv::circle(converted_src, locations_arm[j], 1, CV_RGB(0, 0, 255), 2);

			// depth of current pixel
			float d = (float)ptr[elem_step*locations_arm[j].y + locations_arm[j].x];
			//std::cout << d << std::endl;
			for (size_t k = 0; k < (2 * N_FEAT + 1); k++)
			{
				int idx_x = locations_arm[j].x + (int)(DELTA / d) * ((k - N_FEAT) / N_FEAT);

				for (size_t l = 0; l < (2 * N_FEAT + 1); l++)
				{
					int idx_y = locations_arm[j].y + (int)(DELTA / d) * ((l - N_FEAT) / N_FEAT);
					// read data
					if (idx_x < 0 || idx_x > SRC_COLS || idx_y < 0 || idx_y > SRC_ROWS)
					{
						X[last_ind + i*(N_SAMPLES_0 + N_SAMPLES_1 + N_SAMPLES_2) + N_SAMPLES_1 + j][k*(2 * N_FEAT + 1) + l] = BACKGROUND_DEPTH - d;
						continue;
					}
					if (debug_data)
						cv::circle(converted_src, cv::Point(idx_x, idx_y), 1, CV_RGB(255, 0, 255), 1);
					float d_idx = (float)ptr[elem_step*idx_y + idx_x];
					if (d_idx == 0)
					{
						X[last_ind + i*(N_SAMPLES_0 + N_SAMPLES_1 + N_SAMPLES_2) + N_SAMPLES_1 + j][k*(2 * N_FEAT + 1) + l] = BACKGROUND_DEPTH - d;
						continue;
					}
					X[last_ind + i*(N_SAMPLES_0 + N_SAMPLES_1 + N_SAMPLES_2) + N_SAMPLES_1 + j][k*(2 * N_FEAT + 1) + l] = d_idx - d;

				}
			}
			Y[last_ind + i*(N_SAMPLES_0 + N_SAMPLES_1 + N_SAMPLES_2) + N_SAMPLES_1 + j][0] = 0;

		}


		// extract samples from wrist region
		locations_arm.clear();
		cv::findNonZero(src_Y_ds == 60, locations_wrist);
		// shuffle vector 
		std::random_shuffle(locations_wrist.begin(), locations_wrist.end());

		old_count = locations_wrist.size();
		locations_wrist.resize(2 * old_count);
		std::copy_n(locations_wrist.begin(), old_count, locations_wrist.begin() + old_count);
		old_count = locations_wrist.size();
		locations_wrist.resize(2 * old_count);
		std::copy_n(locations_wrist.begin(), old_count, locations_wrist.begin() + old_count);

		std::cout << locations_wrist.size() << std::endl;

		// display for debugging
		for (size_t j = 0; j < std::min<int>(locations_wrist.size(), N_SAMPLES_2); j++)
		{
			if (debug_data)
				cv::circle(converted_src, locations_wrist[j], 1, CV_RGB(0, 0, 255), 2);

			// depth of current pixel
			float d = (float)ptr[elem_step*locations_wrist[j].y + locations_wrist[j].x];
			//std::cout << d << std::endl;
			for (size_t k = 0; k < (2 * N_FEAT + 1); k++)
			{
				int idx_x = locations_wrist[j].x + (int)(DELTA / d) * ((k - N_FEAT) / N_FEAT);

				for (size_t l = 0; l < (2 * N_FEAT + 1); l++)
				{
					int idx_y = locations_wrist[j].y + (int)(DELTA / d) * ((l - N_FEAT) / N_FEAT);
					// read data
					if (idx_x < 0 || idx_x > SRC_COLS || idx_y < 0 || idx_y > SRC_ROWS)
					{
						X[last_ind + i*(N_SAMPLES_0 + N_SAMPLES_1 + N_SAMPLES_2) + N_SAMPLES_1 + N_SAMPLES_0 + j][k*(2 * N_FEAT + 1) + l] = BACKGROUND_DEPTH - d;
						continue;
					}
					if (debug_data)
						cv::circle(converted_src, cv::Point(idx_x, idx_y), 1, CV_RGB(255, 255, 255), 1);
					float d_idx = (float)ptr[elem_step*idx_y + idx_x];
					if (d_idx == 0)
					{
						X[last_ind + i*(N_SAMPLES_0 + N_SAMPLES_1 + N_SAMPLES_2) + N_SAMPLES_1 + N_SAMPLES_0 + j][k*(2 * N_FEAT + 1) + l] = BACKGROUND_DEPTH - d;
						continue;
					}
					X[last_ind + i*(N_SAMPLES_0 + N_SAMPLES_1 + N_SAMPLES_2) + N_SAMPLES_1 + N_SAMPLES_0 + j][k*(2 * N_FEAT + 1) + l] = d_idx - d;
				}
			}
			Y[last_ind + i*(N_SAMPLES_0 + N_SAMPLES_1 + N_SAMPLES_2) + N_SAMPLES_1 + N_SAMPLES_0 + j][0] = 2;
		}

		if (debug_data)
		{
			double min;
			double max;

			cv::imshow("figaaa", src_Y);
			cv::minMaxIdx(src_Y_ds, &min, &max);
			std::cout << min << std::endl;
			std::cout << max << std::endl;

			cv::imshow("frame", converted_src);
			cv::minMaxIdx(src_X, &min, &max);
			cv::Mat normalized_depth;
			src_X.convertTo(normalized_depth, CV_8UC1, 255.0 / (max - min), -min);
			cv::imshow("depth map", normalized_depth);

			cv::waitKey(0);
		}
	}*/


	std::cout << "data loaded..." << std::endl;
	std::cout << "setting up forest..." << std::endl;	

	uint depth = MAX_DEPTH;
	uint n_trees = N_TREES;

	// let's set up the RDF strucutre for each tree in the forest
	decltype(soil.idecider_vec_t()) cls;
	decltype(soil.ileafmanager_vec_t()) lm;


	for (uint i = 0; i < n_trees; ++i)
	{
		// standard set up feature selector where we need to specify:
		// n_selections_per_node: how many selection proposal are created for each node
		// selection_dimension: how many data dimensions are selected for proposal, must be >0 and < how_many_available
		// how_many_available: how many data dimensions are available
		// max_to_use: how many data dimensions may be used, if we set to zero use how_many_available
		// random_seed
		//auto stdFeatureSelect = soil.StandardFeatureSelectionProvider(1, 10, n_features, n_features, 1 + i);
		//std::cout << n_features << std::endl;
		//auto stdFeatureSelect = soil.StandardFeatureSelectionProvider(10, 1, n_features, n_features, 1 + i);
		auto stdFeatureSelect = soil.StandardFeatureSelectionProvider(N_FEAT_PER_NODE, 1, n_features, n_features, 1 + i);

		// how do you want to combine feature vector into a scalar??

		// calculates a feature as a linear combination of inputs
		// n_params_per_feat_sel: the number of linear configurations to evaluate per feature selection
		// n_comb_dims: the dimensionality of the linear surface. Default: 2.
		// random_seed
		//auto surface = soil.LinearSurfaceCalculator(400, 2, 1 + i);

		// calculates a feature as the difference between two data dimensions of inputs - what I'm gonna need for the project
		//auto surface = soil.DifferenceSurfaceCalculator();

		// select best feature for splitting
		auto surface = soil.AlignedSurfaceCalculator();


		// define information gain at split nodes
		auto shannon = soil.ShannonEntropy();
		auto entropyGain = soil.EntropyGain(shannon);
		// Optimizes a threshold by selecting the best of few random values. It draws n_thresholds random values between the minimum and the maximum observed feature value and returns the best one.
		// n_thresholds 
		// n_classes: number of labels in annotation
		// gain_calculator : The gain calculator IGinCalculator to use
		// gain_threshold: the minimum gain that must be reached to continue splitting
		// annotation_step: The memory step from one annotation value to the next.
		// random_seed: The random seed to initialize the RNG. 
		auto rcto = soil.RandomizedClassificationThresholdOptimizer(100, 3, entropyGain, 0, 1, 1 + i);

		// let's set up the classifier: 
		// \phi: filter function selecting relevant features 
		// \psi: parameters of a function combining features values to single scalar
		// \tau: thresholding parameters for the calculates scalar
		auto tClassifier = soil.ThresholdDecider(stdFeatureSelect, surface, rcto);

		// stores the probability distributions for n_classes at a leaf
		// n_classes: The number of classes used for labeling.
		auto leafMgr = soil.ClassificationLeafManager(3);

		cls.push_back(tClassifier);
		lm.push_back(leafMgr);
	}

	// perform no bagging and use all samples for all trees
	auto nb = soil.NoBagging();

	// Implements the vaniilla decision forest training. Trains all trees independent of each other.
	auto training = soil.ClassicTraining(nb);

	// standard forest class
	// max_tree_depth
	// min_samples_at_leaf
	// min_samples_at_node
	// n_trees
	// deciders
	// leaf manager
	// training
	std::cout << "check" << std::endl;
	auto forest = soil.Forest(depth, 1, 2, n_trees, cls, lm, training);


	// fit the forest to the given data
	std::cout << "fitting to data..." << std::endl;
	forest->fit(X, Y);
	std::cout << "forest fitted" << std::endl;

	//const std::string filename = "C:/Users/remelli/Desktop/Frames/forest.ff";
	const std::string filename = "ff_handsegmentation.ff";
	forest->save(filename);
	}
	
	auto forest = soil.ForestFromFile("ff_handsegmentation.ff");

	if (mode == LIVE)
	{
		std::cout << "starting camera stream" << std::endl;
		// start streaming from the camera
		PXCSenseManager *sense_manager = initialize();


		PXCImage::ImageData depth_buffer;
		PXCImage * sync_depth_pxc;
		PXCCapture::Sample *sample;

		// standard downscaling used by htrack
		int downsampling_factor = 2;
		// downscaling for processing
		int ds = 4;
		cv::Mat sensor_depth = cv::Mat(cv::Size(D_width / downsampling_factor, D_height / downsampling_factor), CV_16UC1, cv::Scalar(0));
		cv::Mat sensor_depth_ds = cv::Mat(cv::Size(D_width / (downsampling_factor*ds), D_height / (downsampling_factor * ds)), CV_16UC1, cv::Scalar(0));

		// TO DO: fix this crap at some point
		std::vector<cv::Point> locations;
		vector<vector< cv::Point> > contours;
		vector<cv::Vec4i> hierarchy;

		while (true) {


			if (sense_manager->AcquireFrame(true) < PXC_STATUS_NO_ERROR) continue;

			sample = sense_manager->QuerySample();

			sample->depth->AcquireAccess(PXCImage::ACCESS_READ_WRITE, PXCImage::PIXEL_FORMAT_DEPTH, &depth_buffer);
			unsigned short* data = ((unsigned short *)depth_buffer.planes[0]);


			/// we downsample the sensor image twice for hand-tracking system, feel free to experiment with the full image as well
			for (int x = 0, x_sub = 0; x_sub < D_width / downsampling_factor; x += downsampling_factor, x_sub++) {
				for (int y = 0, y_sub = 0; y_sub < D_height / downsampling_factor; y += downsampling_factor, y_sub++) {
					sensor_depth.at<unsigned short>(y_sub, x_sub) = data[y* D_width + (D_width - x - 1)];
				}
			}

			sample->depth->ReleaseAccess(&depth_buffer);
			sense_manager->ReleaseFrame();

			cv::medianBlur(sensor_depth, sensor_depth, KERNEL_SIZE);


			cv::resize(sensor_depth, sensor_depth_ds, cv::Size(D_width / (downsampling_factor*ds), D_height / (downsampling_factor * ds)), 0, 0, cv::INTER_NEAREST);


			// show depth map input
			double min;
			double max;
			cv::minMaxIdx(sensor_depth, &min, &max);
			cv::Mat normalized_depth;
			sensor_depth.convertTo(normalized_depth, CV_8UC1, 255.0 / (max - min), -min);
			cv::Mat color_map;
			cv::applyColorMap(normalized_depth, color_map, cv::COLORMAP_COOL);
			cv::imshow("depth", color_map); cv::waitKey(3);

			sensor_depth_ds.setTo(cv::Scalar(BACKGROUND_DEPTH), sensor_depth_ds == 0);
			sensor_depth.setTo(cv::Scalar(BACKGROUND_DEPTH), sensor_depth == 0);


			high_resolution_clock::time_point t1 = high_resolution_clock::now();
			// prepare vector for fast element acces
			sensor_depth_ds.convertTo(src_X, CV_32F);
			float* ptr = (float*)src_X.data;
			size_t elem_step = src_X.step / sizeof(float);

			// build feature vector 
			locations.clear();
			//cv::findNonZero(src_X > 0, locations);

			cv::findNonZero(src_X < GET_CLOSER_TO_SENSOR, locations);
			int n_samples = locations.size();

			
			// allocat memory for new data
			Array<float, 2, 2> new_data = allocate(n_samples, n_features);
			{
				// Extract the lines serially, since the Array class is not thread-safe (yet)
				std::vector<Array<float, 2, 2>::Reference> lines;
				for (int i = 0; i < n_samples; ++i)
				{
					lines.push_back(new_data[i]);
				}
#pragma omp parallel for num_threads(N_THREADS) \
			//default(none) /* Require explicit spec. */\
			shared(ptr,new_data) \
			schedule(static)
				for (int j = 0; j < n_samples; j++)
				{
					// depth of current pixel
					//Array<float, 2, 2> line = allocate(1, n_features);
					std::vector<float> features;
					float d = (float)ptr[elem_step*locations[j].y + locations[j].x];
					for (int k = 0; k < (2 * N_FEAT + 1); k++)
					{
						int idx_x = locations[j].x + (int)(DELTA / d) * ((k - N_FEAT) / N_FEAT);
						for (int l = 0; l < (2 * N_FEAT + 1); l++)
						{
							int idx_y = locations[j].y + (int)(DELTA / d) * ((l - N_FEAT) / N_FEAT);
							// read data
							if (idx_x < 0 || idx_x > SRC_COLS || idx_y < 0 || idx_y > SRC_ROWS)
							{
								features.push_back(BACKGROUND_DEPTH - d);
								continue;
							}
							float d_idx = (float)ptr[elem_step*idx_y + idx_x];
							features.push_back(d_idx - d);
						}
					}
					std::copy(features.begin(), features.end(), lines[j].getData());
				}
			}
			
			// predict data
			Array<double, 2, 2> predictions = forest->predict(new_data, N_THREADS);
			
			// build probability maps for current frame
			// hand
			cv::Mat probabilityMap = cv::Mat::zeros(SRC_ROWS, SRC_COLS, CV_32F);
			for (size_t j = 0; j < locations.size(); j++)
			{
				probabilityMap.at<float>(locations[j]) = predictions[j][1];
			}

			//wrist
			cv::Mat probabilityMap_w = cv::Mat::zeros(SRC_ROWS, SRC_COLS, CV_32F);
			for (size_t j = 0; j < locations.size(); j++)
			{
				probabilityMap_w.at<float>(locations[j]) = predictions[j][2];
			}
			// apply median blur to smooth out along wrist but at the same time keep sharp edges 
			//cv::medianBlur(probabilityMap_w, probabilityMap_w, KERNEL_SIZE);

			// COMPUTE AVERAGE DEPTH OF HAND BLOB ON LOW RES IMAGE

			// threshold low res hand probability map to obtain hand mask
			cv::Mat mask_ds = probabilityMap > THRESHOLD;
			// find biggest blob, a.k.a. hand 
			cv::findContours(mask_ds, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
			int idx = 0, largest_component = 0;
			double max_area = 0;
			for (; idx >= 0; idx = hierarchy[idx][0])
			{
				double area = fabs(cv::contourArea(cv::Mat(contours[idx])));
				if (area > max_area)
				{
					max_area = area;
					largest_component = idx;
				}
			}
			

			// draw biggest blob
			cv::Mat mask_ds_biggest_blob = cv::Mat::zeros(mask_ds.size(), CV_8U);
			cv::drawContours(mask_ds_biggest_blob, contours, largest_component, cv::Scalar(255), CV_FILLED, 8, hierarchy);
			
			// compute average depth
			std::pair<float, int> avg;
			for (int row = 0; row < mask_ds_biggest_blob.rows; ++row)
			{
				for (int col = 0; col < mask_ds_biggest_blob.cols; ++col)
				{
					float depth_wrist = sensor_depth_ds.at<ushort>(row, col);
					if (mask_ds_biggest_blob.at<uchar>(row, col) == 255)
					{
						avg.first += depth_wrist;
						avg.second++;
					}
				}
			}
			ushort depth_hand = (avg.second == 0) ? BACKGROUND_DEPTH : avg.first / avg.second;

			std::cout << "Depth hand:" << std::endl;
			std::cout << depth_hand << std::endl;

			cv::Mat probabilityMap_us;
			cv::Mat probabilityMap_w_us;

			// UPSAMPLE USING RESIZE: advantages of joint bilateral upsampling are already exploited 
			cv::resize(probabilityMap, probabilityMap_us, sensor_depth.size());
			cv::resize(probabilityMap_w, probabilityMap_w_us, sensor_depth.size());
			//jointBilateralUpsample_<float>(probabilityMap, sensor_depth, probabilityMap_us, 6, 6);
			//jointBilateralUpsample_<float>(probabilityMap_w, sensor_depth, probabilityMap_w_us, 6, 6);

			// BUILD HIGH RESOLUTION MASKS FOR HAND AND WRIST
			cv::Mat mask = probabilityMap_us > THRESHOLD;
			cv::Mat mask_wrist = probabilityMap_w_us > THRESHOLD_WRIST;


			// Extract pixels at depth range on hand only
			ushort depth_range = 100;
			cv::Mat range_mask;
			cv::inRange(sensor_depth, depth_hand - depth_range, depth_hand + depth_range, range_mask);

			// POSTPROCESSING: APPLY SOME DILATION and SELECT BIGGEST BLOB
			cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * DILATION_SIZE + 1, 2 * DILATION_SIZE + 1));

			cv::dilate(mask, mask, element);
			mask.setTo(cv::Scalar(120), mask_wrist > 0);
			
			// deep copy because find contours modifies original image

			cv::Mat pp;
			mask.copyTo(pp);

			cv::findContours(pp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
			idx = 0, largest_component = 0;
			max_area = 0;
			for (; idx >= 0; idx = hierarchy[idx][0])
			{
				double area = fabs(cv::contourArea(cv::Mat(contours[idx])));
				//std::cout << area << std::endl;
				if (area > max_area)
				{
					max_area = area;
					largest_component = idx;
				}
			}
			cv::Mat dst = cv::Mat::zeros(mask.size(), CV_8U);
			cv::drawContours(dst, contours, largest_component, cv::Scalar(255), CV_FILLED, 8, hierarchy);
			dst.setTo(cv::Scalar(0), range_mask == 0);
			mask.setTo(cv::Scalar(0), dst == 0);


			// measure performance
			high_resolution_clock::time_point t2 = high_resolution_clock::now();
			std::cout << "TIME FOR SEGMENTING HAND (mS)" << std::endl;
			std::cout << duration_cast<milliseconds> (t2 - t1).count() << std::endl;

			cv::imshow("labelled depth input", mask);
		}
	}

	if (mode == TEST)
	{

		std::cout << "loading data..." << std::endl;
		// read data
		cv::String train_folder = "C:/Users/remelli/Desktop/RDF/Test";
		std::vector<cv::String> filenames_X;
		std::vector<cv::String> filenames_Y;
		cv::glob(train_folder + "/X", filenames_X);
		cv::glob(train_folder + "/Y", filenames_Y);


		// TO DO: fix this crap at some point
		std::vector<cv::Point> locations;
		vector<vector< cv::Point> > contours;
		vector<cv::Vec4i> hierarchy;

		float error = 0.0f;

		for (size_t i = 0; i < SIZE_TEST; i++)
		{
			cv::Mat src_X_ds = cv::Mat(cv::Size(SRC_COLS, SRC_ROWS), CV_16UC1, cv::Scalar(0));
			cv::Mat src_Y_ds = cv::Mat(cv::Size(SRC_COLS, SRC_ROWS), CV_8UC1, cv::Scalar(0));

			std::cout << "Loading frame..." << std::endl;
			std::cout << filenames_X[i] << std::endl;
			// read depth map
			src_X = cv::imread(filenames_X[i], CV_LOAD_IMAGE_UNCHANGED);
			cv::medianBlur(src_X, src_X, KERNEL_SIZE);
			src_X.convertTo(src_X, CV_32F);
			// read ground truth
			std::cout << filenames_Y[i] << std::endl;
			src_Y = cv::imread(filenames_Y[i], CV_LOAD_IMAGE_GRAYSCALE);
			// downsample 
			cv::resize(src_X, src_X_ds, cv::Size(SRC_COLS, SRC_ROWS), 0, 0, cv::INTER_NEAREST);
			cv::resize(src_Y, src_Y_ds, cv::Size(SRC_COLS, SRC_ROWS), 0, 0, cv::INTER_NEAREST);
			//src_X_ds.convertTo(src_X_ds, CV_32F);

			src_X_ds.setTo(cv::Scalar(BACKGROUND_DEPTH), src_X_ds == 0);
			src_X.setTo(cv::Scalar(BACKGROUND_DEPTH), src_X == 0);
			src_Y.setTo(cv::Scalar(0.0), src_Y == 120);
			// prepare vector for fast element acces
			float* ptr = (float*)src_X_ds.data;
			size_t elem_step = src_X_ds.step / sizeof(float);

			// build feature vector 
			locations.clear();
			//cv::findNonZero(src_X > 0, locations);
			cv::findNonZero(src_X_ds != BACKGROUND_DEPTH, locations);
			int n_samples = locations.size();

			// allocat memory for new data
			Array<float, 2, 2> new_data = allocate(n_samples, n_features);
			{
				// Extract the lines serially, since the Array class is not thread-safe (yet)
				std::vector<Array<float, 2, 2>::Reference> lines;
				for (int i = 0; i < n_samples; ++i)
				{
					lines.push_back(new_data[i]);
				}
#pragma omp parallel for num_threads(N_THREADS) \
			//default(none) /* Require explicit spec. */\
			shared(ptr,new_data) \
			schedule(static)
				for (int j = 0; j < n_samples; j++)
				{
					// depth of current pixel
					//Array<float, 2, 2> line = allocate(1, n_features);
					std::vector<float> features;
					float d = (float)ptr[elem_step*locations[j].y + locations[j].x];
					for (int k = 0; k < (2 * N_FEAT + 1); k++)
					{
						int idx_x = locations[j].x + (int)(DELTA / d) * ((k - N_FEAT) / N_FEAT);
						for (int l = 0; l < (2 * N_FEAT + 1); l++)
						{
							int idx_y = locations[j].y + (int)(DELTA / d) * ((l - N_FEAT) / N_FEAT);
							// read data
							if (idx_x < 0 || idx_x > SRC_COLS || idx_y < 0 || idx_y > SRC_ROWS)
							{
								features.push_back(BACKGROUND_DEPTH - d);
								continue;
							}
							float d_idx = (float)ptr[elem_step*idx_y + idx_x];
							features.push_back(d_idx - d);
						}
					}
					std::copy(features.begin(), features.end(), lines[j].getData());
				}
			}

			// predict data
			Array<double, 2, 2> predictions = forest->predict(new_data, N_THREADS);

			// build probability map frame
			cv::Mat probabilityMap = cv::Mat::zeros(SRC_ROWS, SRC_COLS, CV_32F);
			for (size_t j = 0; j < locations.size(); j++)
			{
				probabilityMap.at<float>(locations[j]) = predictions[j][1];
			}

			cv::Mat probabilityMap_w = cv::Mat::zeros(SRC_ROWS, SRC_COLS, CV_32F);
			for (size_t j = 0; j < locations.size(); j++)
			{
				probabilityMap_w.at<float>(locations[j]) = predictions[j][2];
			}


			// COMPUTE AVERAGE DEPTH OF HAND BLOB ON LOW RES IMAGE

			// threshold low res prbability map to mask
			cv::Mat mask_ds = probabilityMap > THRESHOLD;

			// find biggest blob
			cv::findContours(mask_ds, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
			int idx = 0, largest_component = 0;
			double max_area = 0;
			for (; idx >= 0; idx = hierarchy[idx][0])
			{
				double area = fabs(cv::contourArea(cv::Mat(contours[idx])));
				//std::cout << area << std::endl;
				if (area > max_area)
				{
					max_area = area;
					largest_component = idx;
				}
			}
			cv::Mat mask_ds_biggest_blob = cv::Mat::zeros(mask_ds.size(), CV_8U);
			cv::drawContours(mask_ds_biggest_blob, contours, largest_component, cv::Scalar(255), CV_FILLED, 8, hierarchy);
			// compute average
			std::pair<float, int> avg;

			for (int row = 0; row < mask_ds_biggest_blob.rows; ++row)
			{
				for (int col = 0; col < mask_ds_biggest_blob.cols; ++col)
				{
					float depth_wrist = src_X_ds.at<float>(row, col);

					if (mask_ds_biggest_blob.at<uchar>(row, col) == 255)
					{
						avg.first += depth_wrist;
						avg.second++;
					}
				}
			}



			ushort depth_hand = (avg.second == 0) ? BACKGROUND_DEPTH : avg.first / avg.second;
			//std::cout << avg.first << std::endl;
			//std::cout << avg.second << std::endl;
			//std::cout << depth_hand << std::endl;
			// UPSAMPLE USING JOINT BILATERAL FILTER
			cv::Mat probabilityMap_us;
			cv::Mat probabilityMap_w_us;
			//

			cv::resize(probabilityMap, probabilityMap_us, src_X.size());
			cv::resize(probabilityMap_w, probabilityMap_w_us, src_X.size());

			cv::medianBlur(probabilityMap_w_us, probabilityMap_w_us, 5);


			//cv::resize(probabilityMap, probabilityMap_us, cv::Size(D_width / (downsampling_factor), D_height / (downsampling_factor )), 0, 0, cv::INTER_NEAREST);
			
			cv::Mat mask = probabilityMap_us > THRESHOLD;
			cv::Mat mask_wrist = probabilityMap_w_us > THRESHOLD_WRIST;
			//mask.setTo(cv::Scalar(120), probabilityMap_w_us > THRESHOLD_WRIST);



			ushort depth_range = 100;
			cv::Mat range_mask;
			///--- First just extract pixels at the depth range of the wrist
			cv::inRange(src_X, depth_hand - depth_range, /*mm*/
				depth_hand + depth_range, /*mm*/
				range_mask /*=*/);

			//cv::imshow("thresholded hand", mask);


			// POSTPROCESSING: APPLY SOME DILATION and SELECT BIGGEST BLOB
			cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * DILATION_SIZE + 1, 2 * DILATION_SIZE + 1));
			cv::dilate(mask, mask, element);


			//cv::dilate(mask_wrist, mask_wrist, element);
			//cv::erode(mask_wrist, mask_wrist, element);


			//cv::findNonZero(mask_wrist, wrist);
			//cv::RotatedRect wrist_rect = cv::minAreaRect(wrist);
			//std::cout << "min area rect found" << std::endl;

			// We take the edges that OpenCV calculated for us
			//cv::Point2f vertices2f[4];
			//wrist_rect.points(vertices2f);

			// Convert them so we can use them in a fillConvexPoly
			//cv::Point vertices[4];
			//for (int i = 0; i < 4; ++i) {
			//vertices[i] = vertices2f[i];
			//}

			// Now we can fill the rotated rectangle with our specified color
			//cv::fillConvexPoly(mask_wrist,vertices,4,cv::Scalar(120));

			//cv::drawContours(mask_wrist, wrist, 0, cv::Scalar(255), CV_FILLED, 8, hierarchy);


			//cv::rectangle(mask_wrist, wrist_rect, cv::Scalar(130,120,120));


			//cv::imshow("thresholded wrist", mask_wrist);
			mask.setTo(cv::Scalar(60), mask_wrist > 0);

			//cv::imshow("blabla", mask);

			//mask.setTo(cv::Scalar(0), range_mask == 0);

			//cv::imshow("before post processing", mask);
			//cv::dilate(mask, mask, cv::Mat());
			//mask.setTo(cv::Scalar(0), sensor_depth == BACKGROUND_DEPTH);

			cv::Mat pp;
			mask.copyTo(pp);

			cv::findContours(pp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
			idx = 0, largest_component = 0;
			max_area = 0;
			for (; idx >= 0; idx = hierarchy[idx][0])
			{
				double area = fabs(cv::contourArea(cv::Mat(contours[idx])));
				//std::cout << area << std::endl;
				if (area > max_area)
				{
					max_area = area;
					largest_component = idx;
				}
			}
			cv::Mat dst = cv::Mat::zeros(mask.size(), CV_8U);
			cv::drawContours(dst, contours, largest_component, cv::Scalar(255), CV_FILLED, 8, hierarchy);
			dst.setTo(cv::Scalar(0), range_mask == 0);
			mask.setTo(cv::Scalar(0), dst == 0);


			// compute error

			cv::Mat eval_mat = (mask != src_Y);
			std::cout << "Classification error" << std::endl;
			std::cout << (float)cv::countNonZero(eval_mat) / cv::countNonZero(src_Y) * 100 << std::endl;

			error += (float)cv::countNonZero(eval_mat) / cv::countNonZero(src_Y) * 100;

			cv::imshow("RDF prediction", mask);
			cv::imshow("Grund truth", src_Y);
			cv::waitKey(0);


		}

		error = error / SIZE_TEST;

		std::cout << "Average error" << std::endl;
		std::cout << error << std::endl;
	}
	

	return 0;

};