#ifndef GUARD_logging_h
#define GUARD_logging_h

#include "gridinfo.h"

typedef struct {
	PetscLogDouble start;  // moment of the beginnig of the program
	PetscLogDouble prev;  // moment of the previous event
	PetscLogDouble curr;  // moment of the current event
} TimeStamp;

/**
 * initTimeStamp
 * -------------
 * Initialize the time stamp.
 */
PetscErrorCode initTimeStamp(TimeStamp *ts);

/**
 * updateTimeStamp
 * ---------------
 * Update the time stamp, and print it.
 */
PetscErrorCode updateTimeStamp(const VerboseLevel vl, TimeStamp *ts, const char *event_description, const GridInfo gi);

#endif
