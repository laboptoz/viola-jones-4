#include "vj.h"
#include <iomanip>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <numeric>
#include <iostream>
#include <fstream>
#pragma comment(linker, "/STACK:2000000")
#pragma comment(linker, "/HEAP:2000000")


bool StrongClassifier::classify(const cv::Mat& feat_vec) const {

	double strong_result = 0.0;
	double alpha_sum = 0.0;
	for (int i = 0; i < weak_classifiers.size(); ++i) {
		auto weak = weak_classifiers[i];
		auto alpha_i = alphas[i];
		int feature_i = weak.f;
		double actual_val = feat_vec.at<double>(feature_i, 0);
		int p = weak.p;
		double threshold = weak.theta;
		int weak_result = 0;
		//std::cout << actual_val << ", " << threshold << "\n";
		if (p * actual_val <= p * threshold) {
			weak_result = 1;
		}
		else {
			weak_result = 0;
		}

		strong_result += weak_result * alpha_i;
		alpha_sum += 0.5 * alpha_i;
	}

	int strong_classification = 0;
	if (strong_result >= alpha_sum) {
		strong_classification = 1;
	}
	return strong_classification;

}

bool Cascade::classify(const cv::Mat& feature_vector, int stage) const {
	// if one strong classifier is false, classify as false
	int curr_stage = 0;
	for (auto& classifier : strong_classifiers) {
		if (!classifier.classify(feature_vector)) {
			return false;
		}
		if (curr_stage == stage) {
			break;
		}
		curr_stage++;
	}
	return true;
}

void Cascade::write(std::string& filename) const {
	std::ofstream file(filename);
	file << strong_classifiers.size() << "\n";
	for (auto& sc : strong_classifiers) {
		file << sc.weak_classifiers.size() << "\n";
		for (int i = 0; i < sc.weak_classifiers.size(); ++i) {
			file << sc.weak_classifiers[i].f << " ";
			file << sc.weak_classifiers[i].p << " ";
			file << sc.weak_classifiers[i].theta << " ";
			file << sc.alphas[i] << "\n";
		}
	}

}

void Cascade::read(std::string& filename) {
	std::ifstream file(filename);
	strong_classifiers.clear();
	int sc_size;
	file >> sc_size;
	strong_classifiers.resize(sc_size);
	for (int i = 0; i < strong_classifiers.size(); ++i) {
		int wc_size;
		file >> wc_size;
		strong_classifiers[i].weak_classifiers.resize(wc_size);
		strong_classifiers[i].alphas.resize(wc_size);
		for (int k = 0; k < strong_classifiers[i].weak_classifiers.size(); ++k) {
			file >> strong_classifiers[i].weak_classifiers[k].f;
			file >> strong_classifiers[i].weak_classifiers[k].p;
			file >> strong_classifiers[i].weak_classifiers[k].theta;
			file >> strong_classifiers[i].alphas[k];
		}
	}
}

double VJ::get_integral_rect_sum(const cv::Mat& iimg, const cv::Rect& img_rect) const {
	//cv::Mat img_rect = img(rect);
	double vals[4];
	//vals[0] = img_rect.at<double>(0, 0);s
	//vals[1] = img_rect.at<double>(0, img_rect.cols - 1);
	//vals[2] = img_rect.at<double>(img_rect.rows - 1, 0);
	//vals[3] = img_rect.at<double>(img_rect.rows - 1, img_rect.cols - 1);

	vals[0] = iimg.at<double>(img_rect.y, img_rect.x);
	vals[1] = iimg.at<double>(img_rect.y, img_rect.x + img_rect.width);
	vals[2] = iimg.at<double>(img_rect.y + img_rect.height, img_rect.x);
	vals[3] = iimg.at<double>(img_rect.y + img_rect.height, img_rect.x + img_rect.width);

	return vals[3] + vals[0] - (vals[1] + vals[2]);
}

cv::Mat VJ::extract_haar_features(const cv::Mat& img) const {
	// create integral image

	cv::Mat iimg;
	cv::integral(img, iimg, CV_64F);

	cv::Mat fv(166000, 1, CV_64F);
	//cv::Mat fv(300, 1, CV_64F);

	// horizontal
	int k = 0;
	for (int h = 1; h <= 20; ++h) {
		for (int w = 1; w <= 20; ++w) {
			for (int row = 0; row < (iimg.rows - h); ++row) {
				for (int col = 0; col < (iimg.cols - 2 * w); ++col) {
					//std::cout << h << ", " << w << ", " << row << ", " << col << "\n";
					//cv::Mat rect1 = iimg(cv::Rect(col, row, w, h));
					//cv::Mat rect2 = iimg(cv::Rect(col + w, row, w, h));
					cv::Rect rect1 = (cv::Rect(col, row, w, h));
					cv::Rect rect2 = (cv::Rect(col + w, row, w, h));
					double r2 = get_integral_rect_sum(iimg, rect2);
					double r1 = get_integral_rect_sum(iimg, rect1);
					fv.at<double>(k, 0) = r2 - r1;
					k++;
					//if (k == fv.rows) {
					//	return fv;
					//}
				}
			}
		}
	}

	for (int h = 1; h <= 10; ++h) {
		for (int w = 1; w <= 40; ++w) {
			for (int row = 0; row < iimg.rows - 2 * h; ++row) {
				for (int col = 0; col < iimg.cols - w; ++col) {
					//cv::Mat rect1 = iimg(cv::Rect(col, row, w, h));
					//cv::Mat rect2 = iimg(cv::Rect(col, row + h, w, h));
					cv::Rect rect1 = (cv::Rect(col, row, w, h));
					cv::Rect rect2 = (cv::Rect(col, row + h, w, h));
					double r2 = get_integral_rect_sum(iimg, rect2);
					double r1 = get_integral_rect_sum(iimg, rect1);
					fv.at<double>(k, 0) = r1 - r2;
					k++;
				}
			}
		}
	}

	//std::cout << fv;
	return fv;
}


Classifier VJ::select_best_weak_classifier(const std::vector<cv::Mat>& features, 
	std::unordered_map<int, double> wts,
	std::unordered_map<int, int> labels) const {

	int no_of_features = features[0].rows;

	//int no_of_examples = features.size();


	struct ExampleWithI {
		double val;
		int i;
	};

	double Tp = 0.0;
	double Tn = 0.0;
	for (auto& lbl : labels) {
		if (lbl.second == 1) {
			Tp += wts[lbl.first];
		} else {
			Tn += wts[lbl.first];
		}
	}

	//std::vector<double> Tp(no_of_examples, Tp_val);
	//std::vector<double> Tn(no_of_examples, Tn_val);

	double min_error = DBL_MAX;
	Classifier best;


	for (int i = 0; i < no_of_features; ++i) {
		std::vector<ExampleWithI> sorted_examples;
		//sorted_examples.resize(labels.size());
		for (auto& lbl_struct : labels) {
			int example_index = lbl_struct.first;
			ExampleWithI example_with_i;
			example_with_i.i = example_index;
			example_with_i.val = features[example_index].at<double>(i, 0);
			sorted_examples.push_back(example_with_i);
		}

		std::sort(sorted_examples.begin(), sorted_examples.end(), [&](const ExampleWithI& lhs, const ExampleWithI& rhs) {
			return lhs.val < rhs.val;
		});

		//std::vector<double> sorted_weights(no_of_examples);
		//std::vector<int> sorted_labels(no_of_examples);

		//for (int e = 0; e < no_of_examples; ++e) {
		//	sorted_weights[e] = wts[sorted_example_value[e].i];
		//	sorted_labels[e] = labels[sorted_example_value[e].i];
		//}

		// threshold

		double Sn = 0.0;
		double Sp = 0.0;

		for (int e = 0; e < sorted_examples.size(); ++e) {
			int index = sorted_examples[e].i;

			if (labels[index] == 1) {
				// increment positive
				Sp += wts[index];
			} else {
				Sn += wts[index];
			}

			double e_p = Sp + (Tn - Sn);
			double e_n = Sn + (Tp - Sp);

			double curr_min_err = DBL_MAX;
			int p = 0;
			//std::vector<int> result(sorted_examples.size(), 0);
			//std::unordered_map<int, int> result_map;
			if (e_p <= e_n) {
				p = -1;
				curr_min_err = e_p;
				//for (int z = 0; z < sorted_examples.size(); ++z) {
				//	int result_index = sorted_examples[z].i;
				//	if (z > e) {
				//		// greater than current is 1 (positive)
				//		result_map[result_index] = 1;
				//	} else {
				//		// lesser or equal is 0 (negative)
				//		result_map[result_index] = 0;
				//	}
				//}

				//std::fill(result.begin() + e + 1, result.end(), 1);
				//auto result_copy = result;
				//for (int z = 0; z < no_of_examples; ++z) {
				//	result[sorted_example_value[z].i] = result_copy[z];
				//}
			} else {
				p = 1;
				curr_min_err = e_n;
				//for (int z = 0; z < sorted_examples.size(); ++z) {
				//	int result_index = sorted_examples[z].i;
				//	if (z > e) {
				//		// greater than current is 0 (negative)
				//		result_map[result_index] = 0;
				//	} else {
				//		// lesser or equal is 1 (positive)
				//		result_map[result_index] = 1;
				//	}
				//}
				//std::fill(result.begin(), result.begin() + e, 1);
				//auto result_copy = result;
				//for (int z = 0; z < no_of_examples; ++z) {
				//	result[sorted_example_value[z].i] = result_copy[z];
				//}
			}

			if (curr_min_err < min_error) {
				min_error = curr_min_err;
				std::unordered_map<int, int> result_map;
				result_map.reserve(sorted_examples.size());

				if (e_p <= e_n) {
					p = -1;
					curr_min_err = e_p;
					for (int z = 0; z < sorted_examples.size(); ++z) {
						int result_index = sorted_examples[z].i;
						if (z > e) {
							// greater than current is 1 (positive)
							result_map[result_index] = 1;
						}
						else {
							// lesser or equal is 0 (negative)
							result_map[result_index] = 0;
						}
					}

					//std::fill(result.begin() + e + 1, result.end(), 1);
					//auto result_copy = result;
					//for (int z = 0; z < no_of_examples; ++z) {
					//	result[sorted_example_value[z].i] = result_copy[z];
					//}
				}
				else {
					p = 1;
					curr_min_err = e_n;
					for (int z = 0; z < sorted_examples.size(); ++z) {
						int result_index = sorted_examples[z].i;
						if (z > e) {
							// greater than current is 0 (negative)
							result_map[result_index] = 0;
						}
						else {
							// lesser or equal is 1 (positive)
							result_map[result_index] = 1;
						}
					}
					//std::fill(result.begin(), result.begin() + e, 1);
					//auto result_copy = result;
					//for (int z = 0; z < no_of_examples; ++z) {
					//	result[sorted_example_value[z].i] = result_copy[z];
					//}
				}

				best.p = p;

				double t = 0.0;
				if (e == 0) {
					t = sorted_examples[e].val - 0.5;
				} else if (e == sorted_examples.size() - 1) {
					t = sorted_examples[e].val + 0.5;
				} else {
					t = (sorted_examples[e].val + sorted_examples[e + 1].val) / 2.0;
				}
				best.min_error = min_error;

				best.theta = t;
				best.f = i;
				best.result_map = result_map;
			}
		}
	}
	return best;
}

ClassficationResults StrongClassifier::evaluate_classifer(const std::vector<cv::Mat>& features, 
	const std::unordered_map<int, int>& labels,
	std::unordered_map<int, int>& false_positives) {

	false_positives.clear();
	int tp = 0;
	int fp = 0;
	int fn = 0;
	int tn = 0;
	int total_p = 0;
	int total_n = 0;
	for (auto& lbl : labels) {
		int example_i = lbl.first;
		cv::Mat feat_vec = features[example_i];
		bool classification_result = classify(feat_vec);
		if (lbl.second == 1) {
			// positive
			total_p++;
			if (classification_result) {
				tp++;
			}
			else {
				fn++;
			}
		}
		else {
			// negative
			total_n++;
			if (!classification_result) {
				tn++;
			}
			else {
				false_positives[example_i] = 0;
				fp++;
			}
		}
	}

	// true positive %
	double tp_ratio = tp / static_cast<double>(total_p);
	double fp_ratio = fp / static_cast<double>(total_n);
	double tn_ratio = tn / static_cast<double>(total_n);
	double fn_ratio = fn / static_cast<double>(total_p);

	ClassficationResults results;
	results.tp = tp;
	results.fp = fp;
	results.fn = fn;
	results.tn = tn;
	results.tp_ratio = tp_ratio;
	results.tn_ratio = tn_ratio;
	results.fp_ratio = fp_ratio;
	results.fn_ratio = fn_ratio;

	std::cout << "ratios : tp, fp, tn, fn : " << tp_ratio << ", " << fp_ratio << ", " << tn_ratio << ", " << fn_ratio << "\n";
	//std::cout << "vals : tp, fp, tn, fn : " << tp << ", " << fp << ", " << tn << ", " << fn << "\n";

	return results;
}


StrongClassifier VJ::train_with_AdaBoost(const std::vector<cv::Mat>& features, 
	const std::unordered_map<int, int>& positives, 
	const std::unordered_map<int, int>& negatives,
	std::unordered_map<int, int>& new_false_positives) const {

	int total = positives.size() + negatives.size();
	int np = positives.size();
	int nn = negatives.size();

	// init weights
	std::unordered_map<int, double> wts;
	std::unordered_map<int, int> labels;

	double init_pos_wt = 1.0 / (2 * np);
	double init_neg_wt = 1.0 / (2 * nn);

	for (auto& pos : positives) {
		wts[pos.first] = init_pos_wt;
		labels[pos.first] = 1;
	}

	for (auto& neg : negatives) {
		wts[neg.first] = init_neg_wt;
		labels[neg.first] = 0;
	}

	// adaboost iterations
	int T = features[0].rows;

	StrongClassifier strong_classifier;

	for (int t = 0; t < T; ++t) {
		// normalize weights
		double total_wt = 0.0;
		for (auto& wt : wts) {
			total_wt += wt.second;
		}
		for (auto& wt : wts) {
			wt.second /= total_wt;
		}

		// get best feature
		Classifier best_classifier_h_t = select_best_weak_classifier(features, wts, labels);

		double beta_t = best_classifier_h_t.min_error / (1 - best_classifier_h_t.min_error);

		// update weights
		//for (int i = 0; i < wts.size(); ++i) {
		for (auto& wt : wts) {
			int index = wt.first;
			double e_i = best_classifier_h_t.result_map[index] ^ labels[index];
			wt.second = wt.second * std::pow(beta_t, 1 - e_i); 
			//wts[i] = wts[i] * std::pow(beta_t, 1 - e_i);
		}
		//}

		double alpha = DBL_MAX;
		if (beta_t > 1e-8) {
			alpha = std::log(1.0 / beta_t);
		}
		strong_classifier.alphas.push_back(alpha);
		strong_classifier.weak_classifiers.push_back(best_classifier_h_t);

		// evaluate result
		// for this impl, we want true positives to be 100%, and false positives to be 50%
		auto results = strong_classifier.evaluate_classifer(features, labels, new_false_positives);
		if (results.tp_ratio > 0.94 && results.fp_ratio < 0.5) {
			// we are done;
			break;
		}
	}
	return strong_classifier;
}

Cascade VJ::build_cascaded_detector(const std::vector<cv::Mat>& features, const std::unordered_map<int, int>& positives, 
	const std::unordered_map<int, int>& negatives) const {

	std::unordered_map<int, int> positive_map = positives;
	// initialize with true negatives
	std::unordered_map<int, int> false_positives_map = negatives;

	Cascade cascade;

	while (false_positives_map.size() > 0) {
		std::unordered_map<int, int> new_false_positives_map;
		StrongClassifier strong_classifier = train_with_AdaBoost(features, positive_map, false_positives_map, new_false_positives_map);
		std::cout << "No of features : " << strong_classifier.weak_classifiers.size() << "\n";
		std::cout << "Negative samples for next iter : " << new_false_positives_map.size() << "\n";
		false_positives_map = new_false_positives_map;
		cascade.strong_classifiers.push_back(strong_classifier);
	}
	return cascade;


	//double F_target = 1e-10;
	//double F_im1 = 1.0;
	//double D_im1 = 1.0;

	//Imgs P = positives;
	//Imgs N = negatives;


	//double Fi = F_im1;
	//double Di = D_im1;
	//int i = 0;
	//while (Fi > F_target) {
	//	i++;
	//	int ni = 0;
	//	Fi = F_im1;

	//	while (Fi > f * F_im1) {
	//		ni++;
	//		// use P and N to train classifier with ni features
	//		// decrease threshold  for classifer until the detection rate is at least d * D_im1
	//	}

	//	if (Fi > F_target) {
	//		N.clear();
	//		// evaluate current detector and put ANY FALSE detections into the set N
	//		
	//	}
	//}
}

VJ::VJ()
{
}


VJ::~VJ()
{
}

int main(int argc, char** argv) {
	int no_of_images = 100;
	const std::string positive = "positive\\";
	const std::string negative = "negative\\";

	const std::string train = "train\\";
	const std::string test = "test\\";


	std::unordered_map<int, int> positives;
	std::unordered_map<int, int> negatives;
	
	int training_positive_no = 710;
	int training_negative_no = 1758;
	int testing_positive_no = 710;
	int testing_negative_no = 1758;
	//int training_positive_no = 5;
	//int training_negative_no = 5;

	std::vector<cv::Mat>* features = new std::vector<cv::Mat>;

	VJ vj;

	/*
	int cnt = 0;
	for (int i = 0; i < training_positive_no; ++i) {
		std::stringstream ss_train;
		ss_train << std::setfill('0') << std::setw(6);
		ss_train << train << positive << std::setw(6) << (i + 1) << ".png";
		cv::Mat training_img = cv::imread(ss_train.str(), CV_LOAD_IMAGE_GRAYSCALE);
		cv::Mat feature_vec = vj.extract_haar_features(training_img);
		features->push_back(feature_vec);
		positives[cnt] = 1;
		cnt++;
	}

	for (int i = 0; i < training_negative_no; ++i) {
		std::stringstream ss_train;
		ss_train << std::setfill('0') << std::setw(6);
		ss_train << train << negative << std::setw(6) << (i + 1) << ".png";
		cv::Mat training_img = cv::imread(ss_train.str(), CV_LOAD_IMAGE_GRAYSCALE);
		cv::Mat feature_vec = vj.extract_haar_features(training_img);
		features->push_back(feature_vec);
		negatives[cnt] = 0;
		cnt++;
	}

	Cascade cascade = vj.build_cascaded_detector(*features, positives, negatives);


	
	std::string filename = "cascade_trained.txt";
	cascade.write(filename);
*/

	std::string filename = "cascade_trained.txt";
	Cascade cascade;
	cascade.read(filename);


	std::ofstream file("test_results.csv");

	file << "stage," << "fpno, fnno, fp" << "," "fn, tp, tn" << "\n";
	
	for (int stage = 0; stage < cascade.strong_classifiers.size(); ++stage) {

		int tp = 0;
		int fp = 0;
		int tn = 0;
		int fn = 0;
		int total_p = 0;
		int total_n = 0;

		for (int i = 710; i < 888; ++i) {
			std::stringstream ss_train;
			ss_train << std::setfill('0');
			ss_train << test << positive << std::setw(6) << (i + 1) << ".png";
			//std::cout << ss_train.str();
			cv::Mat test_img = cv::imread(ss_train.str(), CV_LOAD_IMAGE_GRAYSCALE);
			cv::Mat feature_vec = vj.extract_haar_features(test_img);
			bool result = cascade.classify(feature_vec, stage);
			total_p++;
			if (result) {
				tp++;
			}
			else {
				fn++;
			}
		}


		for (int i = 1758; i < 2198; ++i) {
			std::stringstream ss_train;
			ss_train << std::setfill('0');
			ss_train << test << negative << std::setw(6) << (i + 1) << ".png";
			cv::Mat test_img = cv::imread(ss_train.str(), CV_LOAD_IMAGE_GRAYSCALE);
			cv::Mat feature_vec = vj.extract_haar_features(test_img);
			bool result = cascade.classify(feature_vec, stage);
			total_n++;
			if (result) {
				fp++;
			}
			else {
				tn++;
			}
		}
		file << stage+1 << "," << fp << "," << fn << "," << (double)fp / total_n << "," << (double)fn / total_p << "," << (double)tp/ total_p << "," <<(double) tn/total_n << "\n";
		std::cout << total_p << " & " << total_n << " & " << tp << " & " << tn << " & " << fp << " & " << fn << " \\\\\n";
	}



	

}