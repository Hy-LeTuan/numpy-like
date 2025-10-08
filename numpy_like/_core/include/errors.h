#ifndef CORE_ERRORS_H_
#define CORE_ERRORS_H_

#define CUSTOM_WARNING_OVERRIDING_CURRENT_DATA                                  \
    "Warning: The current buffer for this ndarray is not empty. Attempting to " \
    "override old data.\n"

#define CUSTOM_ERROR_NOT_A_SEQUENCE                                                    \
    "Sequence Error: Cannot read object as a sequence. A sequence could be a list or " \
    "a tuple.\n"

#define CUSTOM_ERROR_SHAPE_MISMATCH_IN_SEQUENCE_CHILDREN                               \
    "Shape Error: Shape mismatch, children sharing the same level must have the same " \
    "size.\n"

#define CUSTOM_ERROR_DIMENSION_NOT_VALID \
    "Dimension Error: Cannot determine the dimension of the array.\n"

#endif
