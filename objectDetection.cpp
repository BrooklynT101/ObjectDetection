#include "Timer.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <iostream>
#include <string>
#include <limits>


using namespace cv;
using namespace std;

// Global Timer Class
Timer timer;

// Global variables for settings (default values already defined)
std::string objectImageFilename = "object.png";
std::string sceneImageFilename = "scene.png";
std::string detectionMethod = "sift";  // e.g., "sift" or "orb"
std::string matcherMethod = "flann";     // e.g., "flann" or "bfm"

// Global variables
cv::Mat objectImage;
cv::Mat sceneImage;

cv::Mat kptObject;
cv::Mat kptScene;

cv::Mat matchImg;

cv::Ptr<cv::FeatureDetector> detector = cv::SIFT::create();

cv::Ptr<cv::DescriptorMatcher> matcher = cv::FlannBasedMatcher::create();

// Utility function to print current settings, the ones I used for standard output, from the skeleton code
void printUsage() {
	std::cout << "Current settings:" << std::endl;
	std::cout << "Object image: " << objectImageFilename << std::endl;
	std::cout << "Scene image: " << sceneImageFilename << std::endl;
	std::cout << "Detection method: " << detectionMethod << std::endl;
	std::cout << "Matcher method: " << matcherMethod << std::endl;
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
	objectImage = cv::imread(objectImageFilename);
	sceneImage = cv::imread(sceneImageFilename);

	if (objectImage.empty()) {
		std::cerr << "Could not load object image from " << objectImageFilename << std::endl;
		return -1; // Error
	}
	if (sceneImage.empty()) {
		std::cerr << "Could not load scene image from " << sceneImageFilename << std::endl;
		return -1; // Error
	}
	std::cout << "Images loaded successfully, took " << timer.elapsed() << " seconds\n" << std::endl;
	return 0; // Success
}

// Utility function to draw a bounding box around the detected object and save the image under the given filename
int drawBoxAndSaveImage(const std::string& filename, cv::Mat image, std::vector<cv::Point2f>& sceneCorners) {
	// Deep clone the scene image for drawing the box on
	cv::Mat sceneImageClone = image.clone();
	if (sceneImageClone.empty()) {
		std::cerr << "Could not clone the scene image" << std::endl;
		return -1; // Error
	}

	// Draw a bounding box around the detected object using cv::line
	for (size_t i = 0; i < sceneCorners.size(); i++) {
		// Draw lines between corners
		cv::line(sceneImageClone, sceneCorners[i], sceneCorners[(i + 1) % sceneCorners.size()], cv::Scalar(0, 255, 0), 4);
	}
	// Save the detected object
	cv::imwrite(filename, sceneImageClone);
	// Display the result
	//cv::namedWindow(filename, cv::WINDOW_NORMAL);
	//cv::imshow(filename, sceneImageClone);
	return 0;
}

/*
 * Find features in the images, takes in a detector and matcher and assigns the results to the global variables
 */
int findFeatures(cv::Ptr<cv::FeatureDetector> detector, cv::Ptr<cv::DescriptorMatcher> matcher, bool avgTest) {
	// Set up the scoped variables
	std::vector<cv::DMatch> goodMatches;
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	cv::Mat descriptors1, descriptors2;

	// Detect the features in both images
	if (!avgTest) std::cout << "\nDetecting image features..." << std::endl;
	timer.reset();
	detector->detectAndCompute(objectImage, cv::noArray(), keypoints1, descriptors1);
	if (!avgTest) std::cout << "Detected " << keypoints1.size() << " features in object in " << timer.elapsed() << " seconds" << std::endl;
	timer.reset();
	detector->detectAndCompute(sceneImage, cv::noArray(), keypoints2, descriptors2);
	if (!avgTest) std::cout << "Detected " << keypoints2.size() << " features in scene in " << timer.elapsed() << " seconds" << std::endl;

	// Match the descriptors
	if (!avgTest) std::cout << "\nMatching descriptors..." << std::endl;
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
			if (!avgTest) ::cout << "Using ORB detector, setting BFMatching to use Hamming..." << std::endl;
			matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
		}
		timer.reset();
		std::vector<std::vector<cv::DMatch>> matches;
		matcher->knnMatch(descriptors1, descriptors2, matches, 2);
		if (!avgTest) std::cout << "Found " << matches.size() << " matches in " << timer.elapsed() << " seconds using BFM" << std::endl;

		// Filter the matches
		if (!avgTest) std::cout << "\nFinding good matches..." << std::endl;
		timer.reset();
		for (const auto& match : matches) {
			if (match.size() < 2)
				continue;  // Skip if not enough matches are found in this set
			if (match[0].distance < 0.8 * match[1].distance) {
				goodMatches.push_back(match[0]);
			}
		}
		if (!avgTest) std::cout << "Found " << goodMatches.size() << " good matches in " << timer.elapsed() << " seconds" << std::endl;
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
			if (!avgTest) ::cout << "Using ORB detector, setting matchers to use LSH index..." << std::endl;
			cv::Ptr<cv::flann::IndexParams> indexParams = cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2);
			matcher = cv::makePtr<cv::FlannBasedMatcher>(indexParams);
		}

		timer.reset();
		std::vector<std::vector<cv::DMatch>> matches;
		matcher->knnMatch(descriptors1, descriptors2, matches, 2);
		if (!avgTest) std::cout << "Found " << matches.size() << " matches in " << timer.elapsed() << " seconds using FLANN" << std::endl;

		// Filter the matches
		if (!avgTest) std::cout << "\nFinding good matches..." << std::endl;
		timer.reset();
		for (const auto& match : matches) {
			if (match.size() < 2)
				continue;  // Skip if not enough matches are found in this set, solving out of bounds error
			if (match[0].distance < 0.8 * match[1].distance) {
				goodMatches.push_back(match[0]);
			}
		}
		if (!avgTest) std::cout << "Found " << goodMatches.size() << " good matches in " << timer.elapsed() << " seconds" << std::endl;
	}
	else {
		std::cout << "Unknown matcher type" << std::endl;
		return -1; // Error
	}


	//// Use OpenCVs functions to draw keypoints on a copy of the images 
	//if (!avgTest) std::cout << "\nDrawing keypoints..." << std::endl;
	//timer.reset();
	//cv::drawKeypoints(objectImage, keypoints1, kptObject, cv::Scalar(0, 255, 0),
	//	cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//if (!avgTest) std::cout << "Drew keypoints for object in " << timer.elapsed() << " seconds" << std::endl;

	//timer.reset();
	//cv::drawKeypoints(sceneImage, keypoints2, kptScene, cv::Scalar(0, 255, 0),
	//	cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//if (!avgTest) std::cout << "Drew keypoints for scene in " << timer.elapsed() << " seconds" << std::endl;

	//// Draw the matches using OpenCVs function
	//if (!avgTest) std::cout << "\nDrawing matches..." << std::endl;
	//timer.reset();
	//cv::drawMatches(objectImage, keypoints1, sceneImage, keypoints2, goodMatches, matchImg);
	//if (!avgTest) std::cout << "Drawn matches in " << timer.elapsed() << " seconds" << std::endl;
	return 0; // Success
}

/*
 * Run the object detection program using the current settings
 */
int runObjectDetection(bool test) {
	// Load the images even if they are already loaded
	if (loadImages() == -1) return -1;

	// Set up the variables
	std::vector<cv::DMatch> goodMatches;
	std::vector<cv::KeyPoint> objectKeypoints, sceneKeypoints;
	std::vector<cv::Point2f> objectGoodPts, sceneGoodPts;
	std::vector<unsigned char> inliers;
	cv::Mat objectDescriptors, sceneDescriptors;
	cv::Mat detImage = sceneImage.clone(); // Deep copy of the scene image for drawing the box on
	if (detImage.empty()) {
		std::cerr << "Could not clone the scene image" << std::endl;
		return -1; // Error
	}

	// Method info
	if (!test) {
		std::cout << "Running object detection..." << std::endl;
		std::cout << "Object image: " << objectImageFilename << std::endl;
		std::cout << "Scene image: " << sceneImageFilename << std::endl;
		std::cout << "Detection method: " << detectionMethod << std::endl;
		std::cout << "Matcher method: " << matcherMethod << std::endl;
	}

	// Set up the detector based on the settings
	if (toLower(detectionMethod) == "sift") {
		detector = cv::SIFT::create();
	}
	else if (toLower(detectionMethod) == "orb") {
		detector = cv::ORB::create();
	}
	else {
		std::cerr << "Detection method invalid" << std::endl;
		return -1; // Error
	}

	// Set up the matcher based on the settings
	if (toLower(matcherMethod) == "flann") {
		if (dynamic_cast<cv::ORB*>(detector.get())) {
			// If the detector is orb, if so set FLANN matcher to use LSH index
			if (!test) std::cout << "Using ORB detector, setting matchers to use LSH index..." << std::endl;
			// ChatGPT suggested using these parameters for the LSH index to fix an error I was getting, please let me know if this was unnecessary and there was simply something I was missing
			cv::Ptr<cv::flann::IndexParams> indexParams = cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2);
			matcher = cv::makePtr<cv::FlannBasedMatcher>(indexParams);
		}

		// Detect the keypoints and compute the descriptors
		if (!test) std::cout << "\nDetecting image features..." << std::endl;
		timer.reset();
		detector->detectAndCompute(objectImage, cv::Mat(), objectKeypoints, objectDescriptors);
		if (!test) std::cout << "Detected " << objectKeypoints.size() << " features in object in " << timer.elapsed() << " seconds" << std::endl;
		timer.reset();
		detector->detectAndCompute(sceneImage, cv::Mat(), sceneKeypoints, sceneDescriptors);
		if (!test) std::cout << "Detected " << sceneKeypoints.size() << " features in scene in " << timer.elapsed() << " seconds" << std::endl;

		// Draw the KeyPoints on each image and display them
		if (!test) {
			cv::Mat kptObject, kptScene;
			cv::Mat kptObjectImage, kptSceneImage;
			kptObjectImage = objectImage.clone();
			kptSceneImage = sceneImage.clone();
			cv::drawKeypoints(kptObjectImage, objectKeypoints, kptObject, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
			cv::drawKeypoints(kptSceneImage, sceneKeypoints, kptScene, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
			cv::namedWindow("Object Keypoints", cv::WINDOW_NORMAL);
			cv::namedWindow("Scene Keypoints", cv::WINDOW_NORMAL);
			cv::imshow("Object Keypoints", kptObjectImage);
			cv::imshow("Scene Keypoints", kptSceneImage);
		}
		// Match the descriptors
		timer.reset();
		std::vector<std::vector<cv::DMatch>> matches;
		matcher->knnMatch(objectDescriptors, sceneDescriptors, matches, 2);
		if (!test)std::cout << "Found " << matches.size() << " matches in " << timer.elapsed() << " seconds  using FLANN" << std::endl;

		// Filter the matches
		if (!test)std::cout << "\nFinding good matches..." << std::endl;
		timer.reset();
		for (const auto& match : matches) {
			if (match.size() < 2)
				continue;  // Skip if not enough matches are found in this set
			if (match[0].distance < 0.8 * match[1].distance) {
				goodMatches.push_back(match[0]);

				// Extract the good features from the good matches
				objectGoodPts.push_back(objectKeypoints[match[0].queryIdx].pt);
				sceneGoodPts.push_back(sceneKeypoints[match[0].trainIdx].pt);
			}
		}
		if (!test)std::cout << "Found " << goodMatches.size() << " good matches and " << objectGoodPts.size() + sceneGoodPts.size() << " good features from them in " << timer.elapsed() << " seconds" << std::endl;

		// Draw keypoints and good matches
		cv::Mat imgMatches;
		cv::drawMatches(objectImage, objectKeypoints, sceneImage, sceneKeypoints, goodMatches, imgMatches);
	}
	else if (toLower(matcherMethod) == "bfm") {
		// If the detector is ORB, use Hamming distance
		// ChatGPT helped me understand how to detect what detector Im using
		if (dynamic_cast<cv::ORB*>(detector.get())) {
			if (!test)std::cout << "Using ORB detector, setting BFMatching to use Hamming..." << std::endl;
			matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
		}

		// Detect the features
		if (!test)std::cout << "\nDetecting image features..." << std::endl;
		timer.reset();
		detector->detectAndCompute(objectImage, cv::noArray(), objectKeypoints, objectDescriptors);
		if (!test)std::cout << "Detected " << objectKeypoints.size() << " features in object in " << timer.elapsed() << " seconds" << std::endl;
		timer.reset();
		detector->detectAndCompute(sceneImage, cv::noArray(), sceneKeypoints, sceneDescriptors);
		if (!test)std::cout << "Detected " << sceneKeypoints.size() << " features in scene in " << timer.elapsed() << " seconds" << std::endl;

		// Draw the KeyPoints on each image and display them
		cv::Mat kptObject, kptScene;
		cv::Mat kptObjectImage, kptSceneImage;
		if (!test) {
			kptObjectImage = objectImage.clone();
			kptSceneImage = sceneImage.clone();
			cv::drawKeypoints(kptObjectImage, objectKeypoints, kptObject, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
			cv::drawKeypoints(kptSceneImage, sceneKeypoints, kptScene, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
			cv::namedWindow("Object Keypoints", cv::WINDOW_NORMAL);
			cv::namedWindow("Scene Keypoints", cv::WINDOW_NORMAL);
			cv::imshow("Object Keypoints", kptObjectImage);
			cv::imshow("Scene Keypoints", kptSceneImage);
		}

		// Match the descriptors
		timer.reset();
		std::vector<std::vector<cv::DMatch>> matches;
		matcher->knnMatch(objectDescriptors, sceneDescriptors, matches, 2);
		if (!test)std::cout << "Found " << matches.size() << " matches in " << timer.elapsed() << " seconds using BFM" << std::endl;

		// Filter the matches
		if (!test)std::cout << "\nFinding good matches..." << std::endl;
		timer.reset();
		for (const auto& match : matches) {
			if (match.size() < 2)
				continue;  // Skip if not enough matches are found in this set
			if (match[0].distance < 0.8 * match[1].distance) {
				goodMatches.push_back(match[0]);

				// Extract the good features from the good matches
				objectGoodPts.push_back(objectKeypoints[match[0].queryIdx].pt);
				sceneGoodPts.push_back(sceneKeypoints[match[0].trainIdx].pt);
			}
		}
		if (!test)std::cout << "Found " << goodMatches.size() << " good matches and " << objectGoodPts.size() + sceneGoodPts.size() << " good features from them in " << timer.elapsed() << " seconds" << std::endl;
	}
	else {
		std::cerr << "Matcher method invalid" << std::endl;
		return -1; // Error
	}

	// Compute the homography matrix
	timer.reset();
	cv::Mat H = cv::findHomography(objectGoodPts, sceneGoodPts, inliers, cv::RANSAC);
	if (H.empty()) { // Error check
		std::cerr << "Error: Homography matrix is empty. Object not detected!" << std::endl;
		exit(-1);
	}
	if (!test)std::cout << "Calculated the homography matrix in " << timer.elapsed() << " seconds:\n" << H << std::endl;

	// Count the inliers
	int inlierCount = cv::countNonZero(inliers);
	if (!test)std::cout << "Number of inliers: " << inlierCount << " out of " << goodMatches.size() << " good matches." << std::endl;

	// Draw the Bounding Box - used the code from I think lab02 to find the corners
	std::vector<cv::Point2f> objCorners(4);
	objCorners[0] = cv::Point2f(0, 0); // top-left corner
	objCorners[1] = cv::Point2f((float)objectImage.cols, 0); // top-right corner
	objCorners[2] = cv::Point2f((float)objectImage.cols, (float)objectImage.rows); // bottom-right corner
	objCorners[3] = cv::Point2f(0, (float)objectImage.rows); // bottom-left corner

	std::vector<cv::Point2f> sceneCorners(4);
	cv::perspectiveTransform(objCorners, sceneCorners, H);

	if (!test) {
		std::cout << "Transformed Box Coordinates:\n";
		for (const auto& point : sceneCorners) {
			std::cout << point << std::endl;
		}
	}

	// Draw a bounding box around the detected object using cv::line
	for (size_t i = 0; i < sceneCorners.size(); i++) {
		// Draw lines between corners
		cv::line(detImage, sceneCorners[i], sceneCorners[(i + 1) % sceneCorners.size()], cv::Scalar(0, 255, 0), 4);
	}


	if (!test) {
		// Save the detected object
		cv::imwrite("detectedObject.png", detImage);

		// Display the result
		cv::namedWindow("Detection", cv::WINDOW_NORMAL);
		cv::imshow("Detection", detImage);
		// std::cout << "That took " << timer.elapsed() << " seconds" << std::endl;
		cv::waitKey();
	}
	return 0;// Success
}

/*
* Speed test for SIFT and ORB detector, testing time taken for detections using FLANN and then timing again using BFM.
* Average time taken for each test is calculated and displayed at the end of the test, running through each combination
* 20 times and averaging the result.
*/
int speedTest() {
	// Load the images if they are not already loaded
	if (objectImage.empty() || sceneImage.empty()) {
		if (loadImages() == -1) return -1;
	}

	// Defining the feature detector as sift and descriptor matcher as flann at first
	std::cout << "====== Starting SIFT test ======" << std::endl;
	std::cout << "Setting detector to SIFT detector..." << std::endl;
	detector = cv::SIFT::create();
	std::cout << "Setting matcher to FLANN matcher..." << std::endl;
	matcher = cv::FlannBasedMatcher::create(); // Fast Library for Approximate Nearest Neighbours
	findFeatures(detector, matcher, false);

	// Repeating the test using BFM matcher
	std::cout << "\n# Repeating the test using BFM matcher #" << std::endl;
	matcher = cv::BFMatcher::create(); // Brute force matcher
	findFeatures(detector, matcher, false);


	// Running test using ORB detector
	std::cout << "\n\n====== Starting ORB test ======" << std::endl;
	std::cout << "Setting detector to ORB detector..." << std::endl;
	detector = cv::ORB::create();
	std::cout << "Setting matcher to FLANN matcher..." << std::endl;
	matcher = cv::FlannBasedMatcher::create(); // Fast Library for Approximate Nearest Neighbours
	findFeatures(detector, matcher, false);

	// Repeating the test using BFM matcher
	std::cout << "\n# Repeating the test using BFM matcher #" << std::endl;
	matcher = cv::BFMatcher::create(); // Brute force matcher
	findFeatures(detector, matcher, false);

	// Repeating the same test as before, but instead of displaying the results, we will determine the time taken
	// for each combination of detector and matcher and display the average time taken for each test
	std::cout << "\n\n====== Starting feature detection and matching averaging test ======" << std::endl;
	int testCount = 20;
	double totalElapsedTime = 0.0;
	// SIFT detector with FLANN matcher
	detector = cv::SIFT::create();
	matcher = cv::FlannBasedMatcher::create();
	Timer averageTimer = Timer();
	for (int i = 0; i < testCount; i++) {
		averageTimer.reset();
		findFeatures(detector, matcher, true);
		totalElapsedTime += averageTimer.elapsed();
	}
	std::cout << "Average time taken for SIFT detector with FLANN matcher: " << totalElapsedTime / testCount << " seconds" << std::endl;

	// SIFT detector with BFM matcher
	matcher = cv::BFMatcher::create();
	totalElapsedTime = 0.0;
	for (int i = 0; i < testCount; i++) {
		averageTimer.reset();
		findFeatures(detector, matcher, true);
		totalElapsedTime += averageTimer.elapsed();
	}
	std::cout << "Average time taken for SIFT detector with BFM matcher: " << totalElapsedTime / testCount << " seconds" << std::endl;

	// ORB detector with FLANN matcher
	detector = cv::ORB::create();
	matcher = cv::FlannBasedMatcher::create();
	totalElapsedTime = 0.0;
	for (int i = 0; i < testCount; i++) {
		averageTimer.reset();
		findFeatures(detector, matcher, true);
		totalElapsedTime += averageTimer.elapsed();
	}
	std::cout << "Average time taken for ORB detector with FLANN matcher: " << totalElapsedTime / testCount << " seconds" << std::endl;

	// ORB detector with BFM matcher
	matcher = cv::BFMatcher::create();
	totalElapsedTime = 0.0;
	for (int i = 0; i < testCount; i++) {
		averageTimer.reset();
		findFeatures(detector, matcher, true);
		totalElapsedTime += averageTimer.elapsed();
	}
	std::cout << "Average time taken for ORB detector with BFM matcher: " << totalElapsedTime / testCount << " seconds" << std::endl;


	// now run an averaging test with each detector/matcher combination using runObjectDetection
	std::cout << "\n\n====== Starting object detection averaging test ======" << std::endl;
	totalElapsedTime = 0.0;
	// SIFT detector with FLANN matcher
	detector = cv::SIFT::create();
	matcher = cv::FlannBasedMatcher::create();
	for (int i = 0; i < testCount; i++) {
		averageTimer.reset();
		runObjectDetection(true);
		totalElapsedTime += averageTimer.elapsed();
	}
	std::cout << "Average time taken for SIFT detector with FLANN matcher: " << totalElapsedTime / testCount << " seconds" << std::endl;

	// SIFT detector with BFM matcher
	matcher = cv::BFMatcher::create();
	totalElapsedTime = 0.0;
	for (int i = 0; i < testCount; i++) {
		averageTimer.reset();
		runObjectDetection(true);
		totalElapsedTime += averageTimer.elapsed();
	}
	std::cout << "Average time taken for SIFT detector with BFM matcher: " << totalElapsedTime / testCount << " seconds" << std::endl;

	// ORB detector with FLANN matcher
	detector = cv::ORB::create();
	matcher = cv::FlannBasedMatcher::create();
	totalElapsedTime = 0.0;
	for (int i = 0; i < testCount; i++) {
		averageTimer.reset();
		runObjectDetection(true);
		totalElapsedTime += averageTimer.elapsed();
	}
	std::cout << "Average time taken for ORB detector with FLANN matcher: " << totalElapsedTime / testCount << " seconds" << std::endl;

	// ORB detector with BFM matcher
	matcher = cv::BFMatcher::create();
	totalElapsedTime = 0.0;
	for (int i = 0; i < testCount; i++) {
		averageTimer.reset();
		runObjectDetection(true);
		totalElapsedTime += averageTimer.elapsed();
	}
	std::cout << "Average time taken for ORB detector with BFM matcher: " << totalElapsedTime / testCount << " seconds" << std::endl;

	return 0;
}


void detectAndDrawBoundingBox() {
	loadImages();

	// Detect keypoints and compute descriptors using ORB
	cv::Ptr<cv::ORB> orb = cv::ORB::create(50000); // Adjust max features as needed
	std::vector<cv::KeyPoint> objectKeypoints, sceneKeypoints;
	cv::Mat objectDescriptors, sceneDescriptors;
	orb->detectAndCompute(objectImage, cv::Mat(), objectKeypoints, objectDescriptors);
	orb->detectAndCompute(sceneImage, cv::Mat(), sceneKeypoints, sceneDescriptors);

	// FLANN-based matcher for ORB (LshIndexParams is needed for binary descriptors)
	cv::FlannBasedMatcher orbFlannMatcher(
		cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2)
		//cv::makePtr<cv::flann::LshIndexParams>(24, 40, 4)
	);

	// Match descriptors
	std::vector<cv::DMatch> matches;
	orbFlannMatcher.match(objectDescriptors, sceneDescriptors, matches);

	// Find min and max distances between keypoints
	double maxDist = 0;
	double minDist = 100;
	for (const auto& match : matches) {
		double dist = match.distance;
		if (dist < minDist) minDist = dist;
		if (dist > maxDist) maxDist = dist;
	}

	// Filter good matches using Lowe's ratio test
	std::vector<cv::DMatch> goodMatches;
	for (const auto& match : matches) {
		if (match.distance < std::max(2 * minDist, 30.0)) { // 30 is a reasonable ORB threshold
			goodMatches.push_back(match);
		}
	}

	// Extract matched keypoints
	std::vector<cv::Point2f> objPoints, scenePoints;
	for (const auto& match : goodMatches) {
		objPoints.push_back(objectKeypoints[match.queryIdx].pt);
		scenePoints.push_back(sceneKeypoints[match.trainIdx].pt);
	}

	// Compute homography if we have enough points
	if (objPoints.size() >= 4) {
		cv::Mat H = cv::findHomography(objPoints, scenePoints, cv::RANSAC);

		// Get the corners of the object image (rectangle)
		std::vector<cv::Point2f> objCorners = {
			{0, 0},
			{(float)objectImage.cols, 0},
			{(float)objectImage.cols, (float)objectImage.rows},
			{0, (float)objectImage.rows}
		};

		// Warp the corners using the homography matrix
		std::vector<cv::Point2f> sceneCorners(4);
		cv::perspectiveTransform(objCorners, sceneCorners, H);

		// Draw a bounding box around the detected object
		for (int i = 0; i < 4; i++) {
			cv::line(sceneImage, sceneCorners[i], sceneCorners[(i + 1) % 4], cv::Scalar(0, 255, 0), 3);
		}
		// Draw keypoints and good matches
		cv::Mat imgMatches;
		cv::drawMatches(objectImage, objectKeypoints, sceneImage, sceneKeypoints, goodMatches, imgMatches);
		cv::imshow("Keypoints", imgMatches);

		// Display the result
		cv::imshow("Detected Object", sceneImage);
		cv::waitKey(0);
	}
}

/*
* Utility function to get the ground truth corners of the object in the image.
* This is used to calulcate the detection errors for the accuracy test.
*
* The user is prompted to select 4 corners of the object in the image using the mouse,
* in the order of;
* - Top-Left
* - Top-Right
* - Bottom-Right
* - Bottom-Left
*
* ChatGPT was used heavily here as I was unsure how to implement this feature, it helped me understand
* how to use the mouse callback function and how to store the selected points in a vector. The prompt
* was "How can I get the ground truth corners of an object in an image using OpenCV?".
*/
std::vector<cv::Point2f> getGroundTruthCorners(cv::Mat& image) {
	std::vector<cv::Point2f> groundTruthCorners;
	// This is just a shoddy swtich case for default values to save me from selecting them every time
	// you can ignore this and just select the corners manually - just change the boolean
	bool defaultValues = true;
	if (defaultValues) {
		if (sceneImageFilename == "md-scene1.jpg") {
			groundTruthCorners = { { 675, 1736 }, { 1024, 1478 }, { 1328, 1883 }, { 961, 2154 } };
			std::cout << "Using default values for md-scene1.jpg" << std::endl;
			for (const auto& point : groundTruthCorners) {
				std::cout << "Selected corner: (" << point.x << ", " << point.y << ")\n";
			}
			return groundTruthCorners;
		}
		else if (sceneImageFilename == "md-scene2.jpg") {
			groundTruthCorners = { { 961, 1849 }, { 1454, 1430 }, { 1972, 2044 }, { 1471, 2463 } };
			std::cout << "Using default values for md-scene2.jpg" << std::endl;
			for (const auto& point : groundTruthCorners) {
				std::cout << "Selected corner: (" << point.x << ", " << point.y << ")\n";
			}
			return groundTruthCorners;
		}
		else if (sceneImageFilename == "b-scene1.jpg") {
			groundTruthCorners = { { 41, 1576 }, { 978, 1549 }, { 957, 2754 }, { 16, 2800 } };
			std::cout << "Using default values for b-scene1.jpg" << std::endl;
			for (const auto& point : groundTruthCorners) {
				std::cout << "Selected corner: (" << point.x << ", " << point.y << ")\n";
			}
			return groundTruthCorners;
		}
		else if (sceneImageFilename == "b-scene2.jpg") {
			groundTruthCorners = { { 1372, 1085 }, { 2132, 654 }, { 2735, 1638 }, { 1987, 2094 } };
			std::cout << "Using default values for b-scene2.jpg" << std::endl;
			for (const auto& point : groundTruthCorners) {
				std::cout << "Selected corner: (" << point.x << ", " << point.y << ")\n";
			}
			return groundTruthCorners;
		}
		else if (sceneImageFilename == "b-sceneD.jpg") {
			groundTruthCorners = { { 41, 1576 }, { 978, 1549 }, { 957, 2754 }, { 16, 2800 } }; // note that these are the exact same as b-scene1
			std::cout << "Using default values for b-sceneD.jpg" << std::endl;
			for (const auto& point : groundTruthCorners) {
				std::cout << "Selected corner: (" << point.x << ", " << point.y << ")\n";
			}
			return groundTruthCorners;
		}
		else {
			std::cerr << "Scene image not found" << std::endl;
		}
	}
	else {
		auto mouseCallback = [](int event, int x, int y, int flags, void* userdata) {
			auto* corners = reinterpret_cast<std::vector<cv::Point2f>*>(userdata);
			if (event == cv::EVENT_LBUTTONDOWN && corners->size() < 4) {
				corners->emplace_back(x, y);
				std::cout << "Selected corner: (" << x << ", " << y << ")\n";
				if (corners->size() == 4) {
					cv::destroyWindow("Select Object Corners");
				}
			}
			};

		cv::namedWindow("Select Object Corners", cv::WINDOW_NORMAL);
		cv::imshow("Select Object Corners", image);
		cv::setMouseCallback("Select Object Corners", mouseCallback, &groundTruthCorners);
		// Wait until 4 corners are selected
		while (groundTruthCorners.size() < 4) {
			cv::waitKey(10);
		}
		return groundTruthCorners;
	}
}

/*
* Utility function to get the detected corners of the object in the image.
* Almost identical to the runObjectDetection function, but instead of drawing the bounding box,
* it returns the detected corners.
*/
std::vector<cv::Point2f> getDetectedCorners() {
	// Set up the variables
	std::vector<cv::DMatch> goodMatches;
	std::vector<cv::KeyPoint> objectKeypoints, sceneKeypoints;
	std::vector<cv::Point2f> objectGoodPts, sceneGoodPts;
	std::vector<unsigned char> inliers;
	cv::Mat objectDescriptors, sceneDescriptors;
	int nfeaturesCap = 50000; // defining the maximum number of features that each detector can detect up to - default is 500
	double lowesRatio = 0.8; // defines the Lowes Ratio - default is around 0.7 - 0.8

	// Set up the detector based on the settings
	if (toLower(detectionMethod) == "sift") {
		detector = cv::SIFT::create(nfeaturesCap); // default is 500
		matcher = cv::BFMatcher::create(cv::NORM_L2); // though BFM default is NORM_L2, I though I'd specify for clarity
	}
	else if (toLower(detectionMethod) == "orb") {
		/*
		* I need to tweak the ORB settings to get more features, I wasn't able
		* to detect the first image combination so I experimented with the values
		* until the detector was able to accurately detect the object in the scene.
		* I arrived at these values
		*/
		int nf = 50000; // Number of features note: 100,000 is far too much - default is 500
		float sf = 1.2f; // Scale  - default is 1.2
		int nl = 8; // Number of levels - default is 8
		int edgeThreshold = 31; // Edge threshold - default is 31
		detector = cv::ORB::create(nfeaturesCap, sf, nl, edgeThreshold);
		matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
	}
	else {
		std::cerr << "Detection method invalid" << std::endl;
		return {}; // Return empty vector on error
	}

	std::cout << "\nDetecting image features..." << std::endl;
	// Workaround to find the object in md-scene1.jpg
	if (sceneImageFilename == "md-scene1.jpg") {
		// Resize the object image to scale more with the object size in the scene image
		double widthScaleFactor = 0.4; // object in scene is about 0.168x the scale of the object file
		double heightScaleFactor = 0.4; // and for height, it should be around 0.233
		cv::Mat resizedObjectImage;
		cv::resize(objectImage, resizedObjectImage, cv::Size(), widthScaleFactor, heightScaleFactor);
		objectImage = resizedObjectImage; // Update the global objectImage with the resized image
	}
	detector->detectAndCompute(objectImage, cv::noArray(), objectKeypoints, objectDescriptors);
	// Detect object features
	std::cout << "Detected " << objectKeypoints.size() << " features in object." << std::endl;
	// Detect scene features
	detector->detectAndCompute(sceneImage, cv::noArray(), sceneKeypoints, sceneDescriptors);
	std::cout << "Detected " << sceneKeypoints.size() << " features in scene, matching..." << std::endl;

	// Match the descriptors
	std::vector<std::vector<cv::DMatch>> matches;
	matcher->knnMatch(objectDescriptors, sceneDescriptors, matches, 2);
	std::cout << "Found " << matches.size() << " matches using BFM, filtering..." << std::endl;

	// Filter the matches
	for (const auto& match : matches) {
		if (match.size() < 2)
			continue;  // Need two matches to continue
		if (match[0].distance < lowesRatio * match[1].distance) {
			goodMatches.push_back(match[0]);

			// Extract the good features from the good matches
			objectGoodPts.push_back(objectKeypoints[match[0].queryIdx].pt);
			sceneGoodPts.push_back(sceneKeypoints[match[0].trainIdx].pt);
		}
	}
	std::cout << "Found " << goodMatches.size() << " good matches and " << objectGoodPts.size() + sceneGoodPts.size() << " good features from them, computing homography matrix..." << std::endl;

	// Compute the homography matrix
	cv::Mat H = cv::findHomography(objectGoodPts, sceneGoodPts, inliers, cv::RANSAC);
	if (H.empty()) { // Error check
		std::cerr << "Error: Homography matrix is empty. Object not detected!" << std::endl;
		return {}; // Return empty vector on error
	}

	// Count the inliers
	int inlierCount = cv::countNonZero(inliers);
	std::cout << "Number of inliers: " << inlierCount << " out of " << goodMatches.size() << " good matches." << std::endl;

	// Find the Bounding Box
	std::vector<cv::Point2f> objCorners(4);
	objCorners[0] = cv::Point2f(0, 0); // top-left corner
	objCorners[1] = cv::Point2f((float)objectImage.cols, 0); // top-right corner
	objCorners[2] = cv::Point2f((float)objectImage.cols, (float)objectImage.rows); // bottom-right corner
	objCorners[3] = cv::Point2f(0, (float)objectImage.rows); // bottom-left corner

	std::vector<cv::Point2f> sceneCorners(4);
	cv::perspectiveTransform(objCorners, sceneCorners, H);
	// Print the detected corners
	std::cout << "Detected Object Corners:\n";
	for (const auto& point : sceneCorners) {
		std::cout << point << std::endl;
	}
	std::cout << "Returning bounding box.." << std::endl;
	return sceneCorners; // Return the detected corners
}

/*
* Test function to calculate the error in the detected bounding box vs
* the ground truth corners of the object in the image. The error is calculated
* by finding the Euclidean distance between the ground truth corners and the detected
* corners. This function runs through 10 tests, 5 using the ORB detector and 5 using
* the SIFT detector, each using a different scene image. The per corner errors are
* calculated and stored for later, and an average error is displayed at the end of each test.
* At the end of the test the average error for each detector is displayed, including the average
* for each corner in the previous tests. Since we're going for accuracry rather than speed, we can
* use the brute force matcher for this test.
*
* Sorry for the monolithic function, since I needed to hand in the one file I couldn't break
* it up into smaller class files.
*
* ChatGPT helped me understand how to accurately determine and calculate the errors between the
* detected corners and the ground truth corners, the prompt was;
* "How would I accurately determine detection errors between the detected object in scene and the actual corners?"
*/
int accuracyTest() {
	/* Logic Flow
	* For each object and scene test (ten in total, 5 for sift and 5 for orb);
	* - update the global image path variables for object and scene
	* - call loadImages() which updates the global variable holding the 'sceneImage' and 'objectImage'
	* - call the function getGroundTruthCorners(sceneImage) and getting the ground truth corners in return.
	* - run the object detection function with the current settings
	* - calculate the pixel error for the test (including a 5 pixel user error margin), including the average pixel error across all four corners.
	* - display these results, store the per corner errors for later
	* - continue to the next test
	*/

	// ORB detector Tests
	std::cout << "====== Starting ORB accuracy test ======" << std::endl;
	std::vector<double> perCornerErrorsORB(4, 0.0); // Initialize with 0.0
	detectionMethod = "orb";
	matcherMethod = "bfm";
	// ======= Test 1 md.jpg + md-scene1.jpg =======
	std::cout << "======= Test 1: md.jpg + md-scene1.jpg =======" << std::endl;
	// Set the object and scene images and load them
	objectImageFilename = "md.jpg";
	sceneImageFilename = "md-scene1.jpg";
	if (loadImages() == -1) {
		std::cerr << "Error: Could not load images." << std::endl;
		return -1;
	}
	// Get the ground truth corners of the object in the scene image
	std::vector<cv::Point2f> groundTruthCorners = getGroundTruthCorners(sceneImage);
	if (groundTruthCorners.size() != 4) {
		std::cerr << "Error: Ground truth corners not found." << std::endl;
		return -1;
	}
	// Run the object detection function
	std::vector<cv::Point2f> detectedCorners = getDetectedCorners();
	if (detectedCorners.size() != 4) {
		std::cerr << "Error: Detected corners not found." << std::endl;
		return -1;
	}
	// Draw the bounding box around the detected object for debugging
	drawBoxAndSaveImage("detectedObjectORB1.png", sceneImage, detectedCorners);

	// Calculate the pixel error for the test, putting each error into a vector for later, displays the results
	//std::cout << "Calculating pixel error for the test...\n";
	double totalError = 0;
	for (size_t i = 0; i < 4; i++) {
		double error = cv::norm(groundTruthCorners[i] - detectedCorners[i]);
		perCornerErrorsORB[i] += error;
		totalError += error;
		std::cout << "Corner " << i << " Error: " << error << " pixels\n";
	}
	double averageError = totalError / 4.0;
	std::cout << "Average Error: " << averageError << " pixels\n";

	// ======= Test 2 md.jpg + md-scene2.jpg =======
	std::cout << "======= Test 2: md.jpg + md-scene2.jpg =======" << std::endl;
	// set the object and scene images and load them
	objectImageFilename = "md.jpg";
	sceneImageFilename = "md-scene2.jpg";
	if (loadImages() == -1) {
		std::cerr << "Error: Could not load images." << std::endl;
		return -1;
	}
	// Get the ground truth corners of the object in the scene image
	groundTruthCorners = getGroundTruthCorners(sceneImage);
	if (groundTruthCorners.size() != 4) {
		std::cerr << "Error: Ground truth corners not found." << std::endl;
		return -1;
	}
	// Run the object detection function
	detectedCorners = getDetectedCorners();
	if (detectedCorners.size() != 4) {
		std::cerr << "Error: Detected corners not found." << std::endl;
		return -1;
	}
	// Draw the bounding box around the detected object for debugging
	drawBoxAndSaveImage("detectedObjectORB2.png", sceneImage, detectedCorners);
	// Calculate the pixel error for the test, putting each error into a vector for later, displays the results
	totalError = 0;
	for (size_t i = 0; i < 4; i++) {
		double error = cv::norm(groundTruthCorners[i] - detectedCorners[i]);
		perCornerErrorsORB[i] += error;
		totalError += error;
		std::cout << "Corner " << i << " Error: " << error << " pixels\n";
	}
	averageError = totalError / 4.0;
	std::cout << "Average Error: " << averageError << " pixels\n";

	// ======= Test 3 book.jpg + b-scene1.jpg =======
	std::cout << "======= Test 3: book.jpg + b-scene1.jpg =======" << std::endl;
	// set the object and scene images and load them
	objectImageFilename = "book.jpg";
	sceneImageFilename = "b-scene1.jpg";
	if (loadImages() == -1) {
		std::cerr << "Error: Could not load images." << std::endl;
		return -1;
	}
	// Get the ground truth corners of the object in the scene image
	groundTruthCorners = getGroundTruthCorners(sceneImage);
	if (groundTruthCorners.size() != 4) {
		std::cerr << "Error: Ground truth corners not found." << std::endl;
		return -1;
	}
	// Run the object detection function
	detectedCorners = getDetectedCorners();
	if (detectedCorners.size() != 4) {
		std::cerr << "Error: Detected corners not found." << std::endl;
		return -1;
	}
	// Draw the bounding box around the detected object for debugging
	drawBoxAndSaveImage("detectedObjectORB3.png", sceneImage, detectedCorners);

	// Calculate the pixel error for the test, putting each error into a vector for later, displays the results
	totalError = 0;
	for (size_t i = 0; i < 4; i++) {
		double error = cv::norm(groundTruthCorners[i] - detectedCorners[i]);
		perCornerErrorsORB[i] += error;
		totalError += error;
		std::cout << "Corner " << i << " Error: " << error << " pixels\n";
	}
	averageError = totalError / 4.0;
	std::cout << "Average Error: " << averageError << " pixels\n";

	// ======= Test 4 book.jpg + b-scene2.jpg =======
	std::cout << "======= Test 4: book.jpg + b-scene2.jpg =======" << std::endl;
	// set the object and scene images and load them
	objectImageFilename = "book.jpg";
	sceneImageFilename = "b-scene2.jpg";
	if (loadImages() == -1) {
		std::cerr << "Error: Could not load images." << std::endl;
		return -1;
	}
	// Get the ground truth corners of the object in the scene image
	groundTruthCorners = getGroundTruthCorners(sceneImage);
	if (groundTruthCorners.size() != 4) {
		std::cerr << "Error: Ground truth corners not found." << std::endl;
		return -1;
	}
	// Run the object detection function
	detectedCorners = getDetectedCorners();
	if (detectedCorners.size() != 4) {
		std::cerr << "Error: Detected corners not found." << std::endl;
		return -1;
	}
	// Draw the bounding box around the detected object for debugging
	drawBoxAndSaveImage("detectedObjectORB4.png", sceneImage, detectedCorners);
	
	// Calculate the pixel error for the test, putting each error into a vector for later, displays the results
	totalError = 0;
	for (size_t i = 0; i < 4; i++) {
		double error = cv::norm(groundTruthCorners[i] - detectedCorners[i]);
		perCornerErrorsORB[i] += error;
		totalError += error;
		std::cout << "Corner " << i << " Error: " << error << " pixels\n";
	}
	averageError = totalError / 4.0;
	std::cout << "Average Error: " << averageError << " pixels\n";

	// ======= Test 5 book.jpg + b-sceneD.jpg =======
	std::cout << "======= Test 5: book.jpg + b-sceneD.jpg =======" << std::endl;
	// set the object and scene images and load them
	objectImageFilename = "book.jpg";
	sceneImageFilename = "b-sceneD.jpg";
	if (loadImages() == -1) {
		std::cerr << "Error: Could not load images." << std::endl;
		return -1;
	}
	// Get the ground truth corners of the object in the scene image
	groundTruthCorners = getGroundTruthCorners(sceneImage);
	if (groundTruthCorners.size() != 4) {
		std::cerr << "Error: Ground truth corners not found." << std::endl;
		return -1;
	}
	// Run the object detection function
	detectedCorners = getDetectedCorners();
	if (detectedCorners.size() != 4) {
		std::cerr << "Error: Detected corners not found." << std::endl;
		return -1;
	}
	// Draw the bounding box around the detected object for debugging
	drawBoxAndSaveImage("detectedObjectORB5.png", sceneImage, detectedCorners);

	// Calculate the pixel error for the test, putting each error into a vector for later, displays the results
	totalError = 0;
	for (size_t i = 0; i < 4; i++) {
		double error = cv::norm(groundTruthCorners[i] - detectedCorners[i]);
		perCornerErrorsORB[i] += error;
		totalError += error;
		std::cout << "Corner " << i << " Error: " << error << " pixels\n";
	}
	averageError = totalError / 4.0;
	std::cout << "Average Error: " << averageError << " pixels\n";

	// SIFT detector Tests
	std::cout << "====== Starting SIFT accuracy test ======" << std::endl;
	std::vector<double> perCornerErrorsSIFT(4, 0.0); // Initialize with 0.0
	detectionMethod = "sift";
	matcherMethod = "bfm";
	// ======= Test 1 md.jpg + md-scene1.jpg =======
	std::cout << "======= Test 1: md.jpg + md-scene1.jpg =======" << std::endl;
	// Set the object and scene images and load them
	objectImageFilename = "md.jpg";
	sceneImageFilename = "md-scene1.jpg";
	if (loadImages() == -1) {
		std::cerr << "Error: Could not load images." << std::endl;
		return -1;
	}
	// Get the ground truth corners of the object in the scene image
	groundTruthCorners = getGroundTruthCorners(sceneImage);
	if (groundTruthCorners.size() != 4) {
		std::cerr << "Error: Ground truth corners not found." << std::endl;
		return -1;
	}
	// Run the object detection function
	detectedCorners = getDetectedCorners();
	if (detectedCorners.size() != 4) {
		std::cerr << "Error: Detected corners not found." << std::endl;
		return -1;
	}
	// Draw the bounding box around the detected object for debugging
	drawBoxAndSaveImage("detectedObjectSIFT1.png", sceneImage, detectedCorners);

	// Calculate the pixel error for the test, putting each error into a vector for later, displays the results
	totalError = 0;
	for (size_t i = 0; i < 4; i++) {
		double error = cv::norm(groundTruthCorners[i] - detectedCorners[i]);
		perCornerErrorsSIFT[i] += error;
		totalError += error;
		std::cout << "Corner " << i << " Error: " << error << " pixels\n";
	}
	averageError = totalError / 4.0;
	std::cout << "Average Error: " << averageError << " pixels\n";

	// ======= Test 2 md.jpg + md-scene2.jpg =======
	std::cout << "======= Test 2: md.jpg + md-scene2.jpg =======" << std::endl;
	// set the object and scene images and load them
	objectImageFilename = "md.jpg";
	sceneImageFilename = "md-scene2.jpg";
	if (loadImages() == -1) {
		std::cerr << "Error: Could not load images." << std::endl;
		return -1;
	}
	// Get the ground truth corners of the object in the scene image
	groundTruthCorners = getGroundTruthCorners(sceneImage);
	if (groundTruthCorners.size() != 4) {
		std::cerr << "Error: Ground truth corners not found." << std::endl;
		return -1;
	}
	// Run the object detection function
	detectedCorners = getDetectedCorners();
	if (detectedCorners.size() != 4) {
		std::cerr << "Error: Detected corners not found." << std::endl;
		return -1;
	}
	// Draw the bounding box around the detected object for debugging
	drawBoxAndSaveImage("detectedObjectSIFT2.png", sceneImage, detectedCorners);

	// Calculate the pixel error for the test, putting each error into a vector for later, displays the results
	totalError = 0;
	for (size_t i = 0; i < 4; i++) {
		double error = cv::norm(groundTruthCorners[i] - detectedCorners[i]);
		perCornerErrorsSIFT[i] += error;
		totalError += error;
		std::cout << "Corner " << i << " Error: " << error << " pixels\n";
	}
	averageError = totalError / 4.0;
	std::cout << "Average Error: " << averageError << " pixels\n";

	// ======= Test 3 book.jpg + b-scene1.jpg =======
	std::cout << "======= Test 3: book.jpg + b-scene1.jpg =======" << std::endl;
	// set the object and scene images and load them
	objectImageFilename = "book.jpg";
	sceneImageFilename = "b-scene1.jpg";
	if (loadImages() == -1) {
		std::cerr << "Error: Could not load images." << std::endl;
		return -1;
	}
	// Get the ground truth corners of the object in the scene image
	groundTruthCorners = getGroundTruthCorners(sceneImage);
	if (groundTruthCorners.size() != 4) {
		std::cerr << "Error: Ground truth corners not found." << std::endl;
		return -1;
	}
	// Run the object detection function
	detectedCorners = getDetectedCorners();
	if (detectedCorners.size() != 4) {
		std::cerr << "Error: Detected corners not found." << std::endl;
		return -1;
	}
	// Draw the bounding box around the detected object for debugging
	drawBoxAndSaveImage("detectedObjectSIFT3.png", sceneImage, detectedCorners);

	// Calculate the pixel error for the test, putting each error into a vector for later, displays the results
	totalError = 0;
	for (size_t i = 0; i < 4; i++) {
		double error = cv::norm(groundTruthCorners[i] - detectedCorners[i]);
		perCornerErrorsSIFT[i] += error;
		totalError += error;
		std::cout << "Corner " << i << " Error: " << error << " pixels\n";
	}
	averageError = totalError / 4.0;
	std::cout << "Average Error: " << averageError << " pixels\n";

	// ======= Test 4 book.jpg + b-scene2.jpg =======
	std::cout << "======= Test 4: book.jpg + b-scene2.jpg =======" << std::endl;
	// set the object and scene images and load them
	objectImageFilename = "book.jpg";
	sceneImageFilename = "b-scene2.jpg";
	if (loadImages() == -1) {
		std::cerr << "Error: Could not load images." << std::endl;
		return -1;
	}
	// Get the ground truth corners of the object in the scene image
	groundTruthCorners = getGroundTruthCorners(sceneImage);
	if (groundTruthCorners.size() != 4) {
		std::cerr << "Error: Ground truth corners not found." << std::endl;
		return -1;
	}
	// Run the object detection function
	detectedCorners = getDetectedCorners();
	if (detectedCorners.size() != 4) {
		std::cerr << "Error: Detected corners not found." << std::endl;
		return -1;
	}
	// Draw the bounding box around the detected object for debugging
	drawBoxAndSaveImage("detectedObjectSIFT4.png", sceneImage, detectedCorners);

	// Calculate the pixel error for the test, putting each error into a vector for later, displays the results
	totalError = 0;
	for (size_t i = 0; i < 4; i++) {
		double error = cv::norm(groundTruthCorners[i] - detectedCorners[i]);
		perCornerErrorsSIFT[i] += error;
		totalError += error;
		std::cout << "Corner " << i << " Error: " << error << " pixels\n";
	}
	averageError = totalError / 4.0;
	std::cout << "Average Error: " << averageError << " pixels\n";

	// ======= Test 5 book.jpg + b-sceneD.jpg =======
	std::cout << "======= Test 5: book.jpg + b-sceneD.jpg =======" << std::endl;
	// set the object and scene images and load them
	objectImageFilename = "book.jpg";
	sceneImageFilename = "b-sceneD.jpg";
	if (loadImages() == -1) {
		std::cerr << "Error: Could not load images." << std::endl;
		return -1;
	}
	// Get the ground truth corners of the object in the scene image
	groundTruthCorners = getGroundTruthCorners(sceneImage);
	if (groundTruthCorners.size() != 4) {
		std::cerr << "Error: Ground truth corners not found." << std::endl;
		return -1;
	}
	// Run the object detection function
	detectedCorners = getDetectedCorners();
	if (detectedCorners.size() != 4) {
		std::cerr << "Error: Detected corners not found." << std::endl;
		return -1;
	}
	// Draw the bounding box around the detected object for debugging
	drawBoxAndSaveImage("detectedObjectSIFT5.png", sceneImage, detectedCorners);

	// Calculate the pixel error for the test, putting each error into a vector for later, displays the results
	totalError = 0;
	for (size_t i = 0; i < 4; i++) {
		double error = cv::norm(groundTruthCorners[i] - detectedCorners[i]);
		perCornerErrorsSIFT[i] += error;
		totalError += error;
		std::cout << "Corner " << i << " Error: " << error << " pixels\n";
	}
	averageError = totalError / 4.0;
	std::cout << "Average Error: " << averageError << " pixels\n";

	// Calculate the average error for each corner
	std::cout << "====== Average Error for each corner ======" << std::endl;
	std::cout << "ORB Detector:" << std::endl;
	std::cout << "Top-Left Corner: " << perCornerErrorsORB[0] / 5 << " pixels" << std::endl;
	std::cout << "Top-Right Corner: " << perCornerErrorsORB[1] / 5 << " pixels" << std::endl;
	std::cout << "Bottom-Right Corner: " << perCornerErrorsORB[2] / 5 << " pixels" << std::endl;
	std::cout << "Bottom-Left Corner: " << perCornerErrorsORB[3] / 5 << " pixels" << std::endl;
	std::cout << "SIFT Detector:" << std::endl;
	std::cout << "Top-Left Corner: " << perCornerErrorsSIFT[0] / 5 << " pixels" << std::endl;
	std::cout << "Top-Right Corner: " << perCornerErrorsSIFT[1] / 5 << " pixels" << std::endl;
	std::cout << "Bottom-Right Corner: " << perCornerErrorsSIFT[2] / 5 << " pixels" << std::endl;
	std::cout << "Bottom-Left Corner: " << perCornerErrorsSIFT[3] / 5 << " pixels" << std::endl;

	return 0; // Success
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
 * - running accuracy test ~ to compare the accuracy of the SIFT and ORB detector using BFM
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
		std::cout << "5. Accuracy Test\n";
		std::cout << "6. Exit\n";
		std::cout << "Enter your choice (1-5): ";

		if (!(std::cin >> choice)) {
			std::cout << "Invalid input, please input an integer in range of the options.\n";
			clearInputStream();
			continue;
		}

		clearInputStream();

		switch (choice) {
		case 1:
			runObjectDetection(false);
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
			accuracyTest();
			break;
		case 6:
			std::cout << "Exiting program.\n";
			exitProgram = true;
			break;
		case 7:
			detectAndDrawBoundingBox();
			break;
		case 8:
			loadImages();
			getGroundTruthCorners(sceneImage);
			break;
		default:
			std::cout << "Invalid input, please input an integer in range of the options.\n";
			break;
		}
	}

	return 0; // Success
}

int main(int argc, char* argv[]) {
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR); // this was to remove the annoying warning messages and only log actual errors
	int code = -1;
	// Start the main menu
	code = mainMenu();
	return code;
}