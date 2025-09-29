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

static PyTypeObject PyArrayObjectType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0).tp_name = "numpy_like.ndarray",
    .tp_itemsize = 0,
    .tp_basicsize = sizeof(PyArrayObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = PyDoc_STR("ndarray objects"),
    .tp_new = PyType_GenericNew,
};

#endif
