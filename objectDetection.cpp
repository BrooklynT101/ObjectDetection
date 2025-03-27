#include "Timer.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <iostream>
#include <string>

void printUsage() {
	std::cout << "Usage: " << std::endl;
	std::cout << " ObjectDetector <object image> <scene image> <method>" << std::endl;
	std::cout << " <object image> an image of the object to be detected" << std::endl;
	std::cout << " <scene image> an image of a scene to search for the object" << std::endl;
	std::cout << " <method>  SIFT or ORB detection" << std::endl;
	std::cout << " e.g.: ObjectDetector object.png scene.png SIFT" << std::endl;
}

std::string toLower(const std::string& str) {
	std::string result = str;
	std::transform(result.begin(), result.end(), result.begin(),
		[](unsigned char c) { return std::tolower(c); });

	return result;
}

int main(int argc, char* argv[]) {
	Timer timer;

	if (argc != 4) {
		printUsage();
		exit(-1);
	}

	cv::Mat objImage = cv::imread(argv[1]);
	if (objImage.empty()) {
		std::cerr << "Failed to read image from " << argv[1] << std::endl;
		exit(-2);
	}

	cv::Mat scnImage = cv::imread(argv[2]);
	if (scnImage.empty()) {
		std::cerr << "Failed to read image from " << argv[2] << std::endl;
		exit(-3);
	}

	std::string method = toLower(argv[3]);

	if (method != "sift" && method != "orb") {
		std::cerr << "Invalid method '" << argv[3] << "'" << std::endl;
		exit(-4);
	}
	cv::Mat detImage = scnImage.clone();

	///////////////////////////////////////////////////////
	// Code goes here to detect the object in the scene  //
	// You should then draw a box around the object in   //
	// detImage, which has been initialised to be a copy //
	// of the scene.                                     //
	///////////////////////////////////////////////////////

	// Save the detected object
	cv::imwrite("detectedObject.png", detImage);
	cv::namedWindow("Detection");
	cv::imshow("Detection", detImage);
	std::cout << "That took " << timer.elapsed() << " seconds" << std::endl;
	cv::waitKey();
		 
}