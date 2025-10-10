#ifndef CORE_NDARRAY_TYPES_H_
#define CORE_NDARRAY_TYPES_H_

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
};

typedef struct {
} DTYPE_SLOTS;

typedef struct _PyArray_DTypeMeta {
    PyObject_HEAD

    // the same as the PyArrayDescr type_num
    int type_num;

    // slots to functions of this type
    DTYPE_SLOTS *dt_slots;

} PyArray_DTypeMeta;

typedef struct _PyArray_Descr {
    PyObject_HEAD

    /*
     * The type object that poitns to PyArray_DTypeMeta (since this struct is a subclass
     * of the PyTypeObject through the use of PyHeapTypeObject super). The
     * PyArray_DTypeMeta struct has a slots field that is a pointer to the functions
     * that this type suports.
     */
    PyTypeObject *dtype_obj;

    // the single character name for this type in NPY_TYPECHAR
    char type;

    // the integer representation for this type in NPY_TYPES
    int type_num;

    // big or little endian format
    char byteorder;

    // element size (number of bytes) for this type
    npy_intp elsize;
} PyArray_Descr;

typedef struct _PyArrayObject {
    PyObject_HEAD
    char *data;
    int *nd;

    // the size in each dimension, also called shape
    int *dimensions;

    // the amount of byte needed to move to the next element in each dimension
    int *strides;

    // the type descriptor for the current ndarray
    PyArray_Descr *descr;
} PyArrayObject;

extern PyTypeObject PyArrayObjectType;

// Macros

#define NPY_NDIM(arr) *arr->nd
#define NPY_DIM(arr) arr->dimensions
#define NPY_STRIDES(arr) arr->strides
#define NPY_DESCR(arr) arr->descr
#define NPY_ELSIZE(arr) NPY_DESCR(arr)->elsize
#define NPY_DATA(arr) arr->data

#endif
