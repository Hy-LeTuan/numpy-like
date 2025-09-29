#ifndef CORE_NDARRAY_OBJECTS_H_
#define CORE_NDARRAY_OBJECTS_H_

#include <Python.h>

typedef struct _PyArray_Descr {
    PyObject_HEAD PyTypeObject *typeobj;
    char type;
    char byteorder;
} PyArray_Descr;

typedef struct _PyArrayObject {
    PyObject_HEAD char *data;
    int *nd;
    int *dimensions;
    int *strides;
    PyArray_Descr *desc;
} PyArrayObject;

#endif
