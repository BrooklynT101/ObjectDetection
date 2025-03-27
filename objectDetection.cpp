#include "Timer.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <iostream>
#include <string>

Timer timer;

// Global variables
cv::Mat object;
cv::Mat scene;

cv::Mat kptObject;
cv::Mat kptScene;

cv::Mat matchImg;

cv::Ptr<cv::FeatureDetector> detector;
cv::Ptr<cv::DescriptorMatcher> matcher;

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

int loadImages() {
	std::cout << "Loading images..." << std::endl;
	timer.reset();
	object = cv::imread("object.jpg");
	scene = cv::imread("scene.jpg");

	if (object.empty()) {
		std::cerr << "Could not load object from object.jpg" << std::endl;
		return -1; // Error
	}
	if (scene.empty()) {
		std::cerr << "Could not load scene from scene.jpg" << std::endl;
		return -1; // Error
	}
	std::cout << "Images loaded successfully, took " << timer.elapsed() << " seconds" << std::endl;
	return 0; // Success
}

/*
 * Find features in the images, takes in a detector and matcher and assigns the results to the global variables
 */
int findFeatures(cv::Ptr<cv::FeatureDetector> detector, cv::Ptr<cv::DescriptorMatcher> matcher) {
	// Set up the scoped variables
	std::vector<cv::DMatch> goodMatches;
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	cv::Mat descriptors1, descriptors2;

	// Detect the features in both images
	std::cout << "\nDetecting image features..." << std::endl;
	timer.reset();
	detector->detectAndCompute(object, cv::noArray(), keypoints1, descriptors1);
	std::cout << "Detected " << keypoints1.size() << " features in object in " << timer.elapsed() << " seconds" << std::endl;
	timer.reset();
	detector->detectAndCompute(scene, cv::noArray(), keypoints2, descriptors2);
	std::cout << "Detected " << keypoints2.size() << " features in scene in " << timer.elapsed() << " seconds" << std::endl;

	// Match the descriptors
	std::cout << "\nMatching descriptors..." << std::endl;
	timer.reset();

	/**
	* These dynamic cast checks were generated using ChatGPT with the prompt
	* "How can i check if a matcher is a Brute Force matcher in OpenCV?"
	*
	* This just gave me the line that is inside the if statement, but I had to add
	* everything else, I just didnt know how to check what type of detector or matcher
	*/
	// If the matcher is a Brute Force matcher
	if (dynamic_cast<cv::BFMatcher*>(matcher.get())) {
		// If the detector is ORB, use Hamming distance
		if (dynamic_cast<cv::ORB*>(detector.get())) {
			std::cout << "Using ORB detector, setting BFMatching to use Hamming..." << std::endl;
			matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
		}
		timer.reset();
		std::vector<std::vector<cv::DMatch>> matches;
		matcher->knnMatch(descriptors1, descriptors2, matches, 2);
		std::cout << "Found " << matches.size() << " matches in " << timer.elapsed() << " seconds" << std::endl;

		// Filter the matches
		std::cout << "\nFinding good matches..." << std::endl;
		timer.reset();
		for (const auto& match : matches) {
			if (match.size() < 2)
				continue;  // Skip if not enough matches are found in this set
			if (match[0].distance < 0.8 * match[1].distance) {
				goodMatches.push_back(match[0]);
			}
		}
		std::cout << "Found " << goodMatches.size() << " good matches in " << timer.elapsed() << " seconds" << std::endl;
	}
	// If the matcher is a FLANN matcher
	else if (dynamic_cast<cv::FlannBasedMatcher*>(matcher.get())) {
		// If the detector is orb, if so set FLANN matcher to use LSH index
		if (dynamic_cast<cv::ORB*>(detector.get())) {
			/*
			* ChatGPT helped me understand that I can use LSH index with ORB detector,
			* this was after giving it a prompt about an error I was getting with ORB detector.
			* Explained the different types of indexParams and how to use them.
			*/
			std::cout << "Using ORB detector, setting matchers to use LSH index..." << std::endl;
			cv::Ptr<cv::flann::IndexParams> indexParams = cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2);
			matcher = cv::makePtr<cv::FlannBasedMatcher>(indexParams);
		}

		timer.reset();
		std::vector<std::vector<cv::DMatch>> matches;
		matcher->knnMatch(descriptors1, descriptors2, matches, 2);
		std::cout << "Found " << matches.size() << " matches in " << timer.elapsed() << " seconds" << std::endl;

		// Filter the matches
		std::cout << "\nFinding good matches..." << std::endl;
		timer.reset();
		for (const auto& match : matches) {
			if (match.size() < 2)
				continue;  // Skip if not enough matches are found in this set, solving out of bounds error
			if (match[0].distance < 0.8 * match[1].distance) {
				goodMatches.push_back(match[0]);
			}
		}
		std::cout << "Found " << goodMatches.size() << " good matches in " << timer.elapsed() << " seconds" << std::endl;
	}
	else {
		std::cout << "Unknown matcher type" << std::endl;
		return -1; // Error
	}


	// Use OpenCVs functions to draw keypoints on a copy of the images 
	std::cout << "\nDrawing keypoints..." << std::endl;
	timer.reset();
	cv::drawKeypoints(object, keypoints1, kptObject, cv::Scalar(0, 255, 0),
		cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	std::cout << "Drew keypoints for object in " << timer.elapsed() << " seconds" << std::endl;

	timer.reset();
	cv::drawKeypoints(scene, keypoints2, kptScene, cv::Scalar(0, 255, 0),
		cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	std::cout << "Drew keypoints for scene in " << timer.elapsed() << " seconds" << std::endl;

	// Draw the matches using OpenCVs function
	std::cout << "\nDrawing matches..." << std::endl;
	timer.reset();
	cv::drawMatches(object, keypoints1, scene, keypoints2, goodMatches, matchImg);
	std::cout << "Drawn matches in " << timer.elapsed() << " seconds" << std::endl;
	return 0; // Success
}

/*
Speed test for SIFT and ORB detector, testing time taken for detections using FLANN and then timing again using BFM
*/
int speedTest() {
	// Load the images if they are not already loaded
	if (object.empty() || scene.empty()) {
		if (loadImages() == -1) return -1;
	}

	// Defining the feature detector as sift and descriptor matcher as flann at first
	std::cout << "====== Starting SIFT test ======" << std::endl;
	std::cout << "Setting detector to SIFT detector..." << std::endl;
	detector = cv::SIFT::create();
	std::cout << "Setting matcher to FLANN matcher..." << std::endl;
	matcher = cv::FlannBasedMatcher::create(); // Fast Library for Approximate Nearest Neighbours
	findFeatures(detector, matcher);

	// Repeating the test using BFM matcher
	std::cout << "\n# Repeating the test using BFM matcher #" << std::endl;
	matcher = cv::BFMatcher::create(); // Brute force matcher
	findFeatures(detector, matcher);


	// Running test using ORB detector
	std::cout << "\n\n====== Starting ORB test ======" << std::endl;
	std::cout << "Setting detector to ORB detector..." << std::endl;
	detector = cv::ORB::create();
	std::cout << "Setting matcher to FLANN matcher..." << std::endl;
	matcher = cv::FlannBasedMatcher::create(); // Fast Library for Approximate Nearest Neighbours
	findFeatures(detector, matcher);

	// Repeating the test using BFM matcher
	std::cout << "\n# Repeating the test using BFM matcher #" << std::endl;
	matcher = cv::BFMatcher::create(); // Brute force matcher
	findFeatures(detector, matcher);
	return 0;
}

/*
 * Settings menu for the object detection program
 * 
 * Includes options for: 
 * - defining object and scene images ~ user can define the image filenames for scene and object or use default filenames (object.jpg and scene.jpg)
 * - choosing detection method ~ choice SIFT or ORB, then FLANN or BFM matcher, then returns to the main menu
 */
int settingsMenu() {
	return 0; // Success
}

/*
 * User terminal for the object detection program
 *
 * Includes options for:
 * - Running object detection ~ runs the standard assignment test using the current settings
 * - print usage ~ to display the current settings the program is using
 * - settings menu ~ to change the object and scene images or the detection method
 * - running speed test ~ to compare SIFT and ORB detector with FLANN and BFM matcher
 * - exit ~ to close the program
 */
int mainMenu() {
	return 0; // Success
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