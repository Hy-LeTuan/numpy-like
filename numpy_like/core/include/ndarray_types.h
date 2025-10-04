#ifndef CORE_NDARRAY_OBJECTS_H_
#define CORE_NDARRAY_OBJECTS_H_

#include <Python.h>

#define NPY_MAXDIMS 6
typedef Py_ssize_t npy_intp;

enum NPY_TYPES {
    NPY_BOOL = 0,
    NPY_BYTE = 1,
    NPY_UBYTE = 2,
    NPY_SHORT = 3,
    NPY_USHORT = 4,
    NPY_INT = 5,
    NPY_UINT = 6,
    NPY_LONG = 7,
    NPY_ULONG = 8,
    NPY_LONGLONG = 9,
    NPY_ULONGLONG = 10,
    NPY_FLOAT = 11,
    NPY_DOUBLE = 12,
    NPY_LONGDOUBLE = 13,
    NPY_CFLOAT = 14,
    NPY_CDOUBLE = 15,
    NPY_CLONGDOUBLE = 16,
    NPY_OBJECT = 17,
    NPY_STRING = 18,
};

enum NPY_TYPECHAR {
    NPY_BOOLLTR = '?',
    NPY_BYTELTR = 'b',
    NPY_UBYTELTR = 'B',
    NPY_SHORTLTR = 'h',
    NPY_USHORTLTR = 'H',
    NPY_INTLTR = 'i',
    NPY_UINTLTR = 'I',
    NPY_LONGLTR = 'l',
    NPY_ULONGLTR = 'L',
    NPY_LONGLONGLTR = 'q',
    NPY_ULONGLONGLTR = 'Q',
    NPY_FLOATLTR = 'f',
    NPY_DOUBLELTR = 'd',
    NPY_LONGDOUBLELTR = 'g',
    NPY_CFLOATLTR = 'F',
    NPY_CDOUBLELTR = 'D',
    NPY_CLONGDOUBLELTR = 'G',
    NPY_OBJECTLTR = 'O',
    NPY_STRINGLTR = 'S',
};

typedef struct _PyArray_Descr {
    PyObject_HEAD

    /*
     * The type representation of this type that is exposed to Python
     */
    PyTypeObject *typeobj;

    // the single character name for this type in NPY_TYPECHAR
    char type;

    // the integer representation for this type in NPY_TYPES
    int type_num;

    char byteorder;

    // element size (itemsize) for this type
    npy_intp elsize;
} PyArray_Descr;

typedef struct _PyArrayObject {
    PyObject_HEAD
    char *data;
    int *nd;
    int *dimensions;
    int *strides;
    PyArray_Descr *desc;
} PyArrayObject;

extern PyTypeObject PyArrayObjectType;

#endif
