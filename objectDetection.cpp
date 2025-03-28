#include "Timer.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <iostream>
#include <string>
#include <limits>

// Global Timer Class
Timer timer;

// Global variables for settings (default values already defined)
std::string objectImageFilename = "object.jpg";
std::string sceneImageFilename = "scene.jpg";
std::string detectionMethod = "sift";  // e.g., "sift" or "orb"
std::string matcherMethod = "flann";     // e.g., "flann" or "bfm"

// Global variables
cv::Mat object;
cv::Mat scene;

cv::Mat kptObject;
cv::Mat kptScene;

cv::Mat matchImg;

cv::Ptr<cv::FeatureDetector> detector;
cv::Ptr<cv::DescriptorMatcher> matcher;

/*
* Print the usage of the program, to show which values are loaded for the object detection
*/
void printUsage() {
	std::cout << "Usage: " << std::endl;
	std::cout << " ObjectDetector <object image> <scene image> <method>" << std::endl;
	std::cout << " <object image> an image of the object to be detected" << std::endl;
	std::cout << " <scene image> an image of a scene to search for the object" << std::endl;
	std::cout << " <method>  SIFT or ORB detection" << std::endl;
	std::cout << " e.g.: ObjectDetector object.png scene.png SIFT" << std::endl;
}

// Utility function to convert a string to lowercase
std::string toLower(const std::string& str) {
	std::string result = str;
	std::transform(result.begin(), result.end(), result.begin(),
		[](unsigned char c) { return std::tolower(c); });

	return result;
}

// Utility function to clear std::cin error flags and ignore the rest of the line.
void clearInputStream() {
	std::cin.clear();
	std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

// Utility function to load images from the filenames
int loadImages() {
	std::cout << "Loading images..." << std::endl;
	timer.reset();
	object = cv::imread(objectImageFilename);
	scene = cv::imread(sceneImageFilename);

	if (object.empty()) {
		std::cerr << "Could not load object image from " << objectImageFilename << std::endl;
		return -1; // Error
	}
	if (scene.empty()) {
		std::cerr << "Could not load scene image from " << sceneImageFilename << std::endl;
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
 * Run the object detection program using the current settings
 */
int runObjectDetection() {
	return 0;// Success
}

/*
 * Settings menu for the object detection program
 *
 * Includes options for:
 * - defining object and scene images ~ user can define the image filenames for scene and object or use default filenames (object.jpg and scene.jpg)
 * - choosing detection method ~ choice SIFT or ORB, then FLANN or BFM matcher, then returns to the main menu
 */
int settingsMenu() {
	int choice = 0;
	bool exitMenu = false;

	while (!exitMenu) {
		std::cout << "\n--- Settings Menu ---\n";
		std::cout << "1. Set object image filename (current: " << objectImageFilename << ")\n";
		std::cout << "2. Set scene image filename (current: " << sceneImageFilename << ")\n";
		std::cout << "3. Set detection method (current: " << detectionMethod << ")\n";
		std::cout << "4. Set matcher method (current: " << matcherMethod << ")\n";
		std::cout << "5. Return to Main Menu\n";
		std::cout << "Enter your choice (1-5): ";

		if (!(std::cin >> choice)) {
			std::cout << "Invalid input, please input an integer in range of the options.\n";
			clearInputStream();
			continue;
		}

		clearInputStream(); // clear leftover newline characters

		switch (choice) {
		case 1: {
			std::cout << "Enter new object image filename: ";
			std::getline(std::cin, objectImageFilename);
			std::cout << "Object image filename set to: " << objectImageFilename << "\n";
			break;
		}
		case 2: {
			std::cout << "Enter new scene image filename: ";
			std::getline(std::cin, sceneImageFilename);
			std::cout << "Scene image filename set to: " << sceneImageFilename << "\n";
			break;
		}
		case 3: {
			std::cout << "Choose detection method (sift/orb): ";
			std::string input;
			std::getline(std::cin, input);
			// Convert input to lowercase
			std::transform(input.begin(), input.end(), input.begin(), ::tolower);
			if (input == "sift" || input == "orb") {
				detectionMethod = input;
				std::cout << "Detection method set to: " << detectionMethod << "\n";
			}
			else {
				std::cout << "Invalid detection method. Please choose either 'sift' or 'orb'.\n";
			}
			break;
		}
		case 4: {
			std::cout << "Choose matcher method (flann/bfm): ";
			std::string input;
			std::getline(std::cin, input);
			// Convert input to lowercase
			std::transform(input.begin(), input.end(), input.begin(), ::tolower);
			if (input == "flann" || input == "bfm") {
				matcherMethod = input;
				std::cout << "Matcher method set to: " << matcherMethod << "\n";
			}
			else {
				std::cout << "Invalid matcher method. Please choose either 'flann' or 'bfm'.\n";
			}
			break;
		}
		case 5: {
			exitMenu = true;
			break;
		}
		default: {
			std::cout << "Invalid input, please input an integer in range of the options.\n";
			break;
		}
		}
	}
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
 *
 * Had ChatGPT help me understand how to structure the main menu and read user input,
 * then used the same structure for the settings menu. Returned some broken code I had to alter
 * to work with the existing program.
 */
int mainMenu() {
	int choice = 0;
	bool exitProgram = false;

	while (!exitProgram) {
		std::cout << "\n--- Main Menu ---\n";
		std::cout << "1. Run Object Detection\n";
		std::cout << "2. Print Usage (Current Settings)\n";
		std::cout << "3. Settings Menu\n";
		std::cout << "4. Speed Test\n";
		std::cout << "5. Exit\n";
		std::cout << "Enter your choice (1-5): ";

		if (!(std::cin >> choice)) {
			std::cout << "Invalid input, please input an integer in range of the options.\n";
			clearInputStream();
			continue;
		}

		clearInputStream();

		switch (choice) {
		case 1:
			runObjectDetection();
			break;
		case 2:
			printUsage();
			break;
		case 3:
			settingsMenu();
			break;
		case 4:
			speedTest();
			break;
		case 5:
			std::cout << "Exiting program.\n";
			exitProgram = true;
			break;
		default:
			std::cout << "Invalid input, please input an integer in range of the options.\n";
			break;
		}
	}

	return 0; // Success
}

int main(int argc, char* argv[]) {
	// Start the main menu
	mainMenu();

	///////////////////////////////////////////////////////
	// Code goes here to detect the object in the scene  //
	// You should then draw a box around the object in   //
	// detImage, which has been initialised to be a copy //
	// of the scene.                                     //
	///////////////////////////////////////////////////////
}