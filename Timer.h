#pragma once

#include <chrono>

class Timer {

public:

	// Create and start a new Timer
	Timer();

	// Copy construction and assignment
	Timer(const Timer&) = default;
	~Timer() = default;

	// Destructor
	Timer& operator=(const Timer&) = default;

	// Reset the Timer start to now
	void reset();

	// Return the elapsed time since starting (or last reset) in seconds
	// Time is given to millisecond precision (3 decimal places)
	double elapsed() const;

private:

	std::chrono::time_point<std::chrono::high_resolution_clock> start;
};
