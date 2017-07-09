#pragma once
#include <vector>
#include "opencv2/core/core.hpp"
#include <unordered_map>

using FV = std::vector < double > ;
using Imgs = std::vector < cv::Mat > ;

struct ClassficationResults {
	int tp;
	int fp;
	int fn;
	int tn;
	int total_p;
	int total_n;
	
	double tp_ratio;
	double fp_ratio;
	double tn_ratio;
	double fn_ratio;
};

struct Classifier {
	int p;
	int f;
	double theta;
	double min_error;
	//std::vector<int> result;
	std::unordered_map<int, int> result_map;

};

struct StrongClassifier {

	std::vector<double> alphas;
	std::vector<Classifier> weak_classifiers;
	
	bool classify(const cv::Mat& feature_vector) const;
	ClassficationResults evaluate_classifer(const std::vector<cv::Mat>& features, const std::unordered_map<int, int>& labels, std::unordered_map<int, int>& false_positives);
};

struct Cascade {
	std::vector<StrongClassifier> strong_classifiers;
	bool classify(const cv::Mat& feature_vector, int stage) const;
	void write(std::string& filename) const;
	void read(std::string& filename);
};

class VJ
{
public:
	double get_integral_rect_sum(const cv::Mat& iimg, const cv::Rect& img_rect) const;
	//double get_integral_rect_sum(const cv::Mat& img_rect) const;
	cv::Mat extract_haar_features(const cv::Mat& img) const;

	//cv::Mat create_integral_img(const cv::Mat& img) const;

	Classifier select_best_weak_classifier(const std::vector<cv::Mat>& features, std::unordered_map<int, double> wts, std::unordered_map<int, int> labels) const;

	StrongClassifier train_with_AdaBoost(const std::vector<cv::Mat>& features, 
		const std::unordered_map<int, int>& positives, 
		const std::unordered_map<int, int>& negatives,
		std::unordered_map<int, int>& false_positives) const;

	Cascade build_cascaded_detector(const std::vector<cv::Mat>& features, 
		const std::unordered_map<int, int>& positives, 
		const std::unordered_map<int, int>& negatives) const;

	VJ();
	~VJ();
};

