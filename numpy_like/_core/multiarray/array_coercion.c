#include <Python.h>
#include <errors.h>
#include <ndarray_types.h>

#include "boolobject.h"

PyArray_Descr *
PyArray_CreatePyArrayDescr(enum NPY_TYPES type)
{
    PyArray_Descr *out = malloc(sizeof(PyArray_Descr));

    switch (type) {
        case NPY_BOOL:
            out->typeobj = NULL;
            out->type = '?';
            out->type_num = (int)NPY_BOOL;
            out->byteorder = 'B';
            out->elsize = 1;
            break;
        case NPY_BYTE:
            out->typeobj = NULL;
            out->type = 'b';
            out->type_num = (int)NPY_BYTE;
            out->byteorder = 'B';
            out->elsize = 1;
            break;
        case NPY_UBYTE:
            out->typeobj = NULL;
            out->type = 'B';
            out->type_num = (int)NPY_UBYTE;
            out->byteorder = 'B';
            out->elsize = 1;
            break;
        case NPY_SHORT:
            out->typeobj = NULL;
            out->type = 'B';
            out->type_num = (int)NPY_SHORT;
            out->byteorder = 'h';
            out->elsize = 2;
            break;
        case NPY_USHORT:
            out->typeobj = NULL;
            out->type = 'B';
            out->type_num = (int)NPY_USHORT;
            out->byteorder = 'H';
            out->elsize = 2;
            break;
        case NPY_INT:
            out->typeobj = NULL;
            out->type = 'B';
            out->type_num = (int)NPY_USHORT;
            out->byteorder = 'i';
            out->elsize = 4;
            break;
        case NPY_UINT:
            out->typeobj = NULL;
            out->type = 'B';
            out->type_num = (int)NPY_UINT;
            out->byteorder = 'I';
            out->elsize = 4;
            break;
        case NPY_LONG:
            out->typeobj = NULL;
            out->type = 'B';
            out->type_num = (int)NPY_LONG;
            out->byteorder = 'l';
            out->elsize = 8;
            break;
        case NPY_ULONG:
            out->typeobj = NULL;
            out->type = 'B';
            out->type_num = (int)NPY_ULONG;
            out->byteorder = 'L';
            out->elsize = 8;
            break;
        case NPY_LONGLONG:
            out->typeobj = NULL;
            out->type = 'B';
            out->type_num = (int)NPY_LONGLONG;
            out->byteorder = 'q';
            out->elsize = 8;
            break;
        case NPY_ULONGLONG:
            out->typeobj = NULL;
            out->type = 'B';
            out->type_num = (int)NPY_ULONGLONG;
            out->byteorder = 'Q';
            out->elsize = 8;
            break;
        case NPY_FLOAT:
            out->typeobj = NULL;
            out->type = 'B';
            out->type_num = (int)NPY_FLOAT;
            out->byteorder = 'f';
            out->elsize = 4;
            break;
        case NPY_DOUBLE:
            out->typeobj = NULL;
            out->type = 'B';
            out->type_num = (int)NPY_DOUBLE;
            out->byteorder = 'd';
            out->elsize = 8;
            break;
        case NPY_LONGDOUBLE:
            out->typeobj = NULL;
            out->type = 'B';
            out->type_num = (int)NPY_LONGDOUBLE;
            out->byteorder = 'g';
            out->elsize = 16;
            break;
        case NPY_CFLOAT:
            out->typeobj = NULL;
            out->type = 'B';
            out->type_num = (int)NPY_CFLOAT;
            out->byteorder = 'F';
            out->elsize = 4;
            break;
        case NPY_CDOUBLE:
            out->typeobj = NULL;
            out->type = 'B';
            out->type_num = (int)NPY_CDOUBLE;
            out->byteorder = 'D';
            out->elsize = 8;
            break;
        case NPY_CLONGDOUBLE:
            out->typeobj = NULL;
            out->type = 'B';
            out->type_num = (int)NPY_CLONGDOUBLE;
            out->byteorder = 'G';
            out->elsize = 16;
            break;
    }

    return out;
}

/*
 * This is going to get called when we reach indivual elements in the original
 * sequence. Assume that this function is only called with out_descr = NULL
 * @param obj:
 * @returns Returns 1 if the function can populate out_descr and 0 otherwise.
 * */
int
PyArray_DeterminePyArrayDescr(PyObject *obj, PyArray_Descr **out_descr)
{
    if (*out_descr != NULL)
        return 0;

    PyArray_Descr *out;

    if (PyLong_Check(obj)) {
        out = PyArray_CreatePyArrayDescr(NPY_INT);
    }
    else if (PyFloat_Check(obj)) {
        out = PyArray_CreatePyArrayDescr(NPY_FLOAT);
    }
    else if (PyBool_Check(obj)) {
        out = PyArray_CreatePyArrayDescr(NPY_BOOL);
    }
    else {
        // cannot determine type
        return 0;
    }

    *out_descr = out;

    return 1;
}

/**
 * @returns Returns -1 if shape mismatch happens
 * */
static int
PyArray_UpdateShape(npy_intp out_shape[NPY_MAXDIMS], int size, int curr_dim)
{
    // shape mismatch since all elements of the same depth should have the same size
    if (out_shape[curr_dim - 1] > 0 && out_shape[curr_dim - 1] != size) {
        PyErr_SetString(PyExc_RuntimeError,
                        CUSTOM_ERROR_SHAPE_MISMATCH_IN_SEQUENCE_CHILDREN);
        return -1;
    }

    if (out_shape[curr_dim - 1] == 0)
        out_shape[curr_dim - 1] = size;

    return 0;
}

/**
 * @param curr_dim: Current dimension, starts at 1 and count up to NPY_MAXDIMS
 * inclusive.
 * **/
int
PyArray_DiscoverDTYpeAndShapeRecursive(PyObject *obj, int max_dims, int curr_dim,
                                       npy_intp out_shape[NPY_MAXDIMS],
                                       PyArray_Descr **out_descr)
{
    if (!PySequence_Check(obj)) {
        if (*out_descr == NULL) {
            PyArray_DeterminePyArrayDescr(obj, out_descr);
        }

        return curr_dim - 1;
    }
    else if (curr_dim > max_dims) {
        return -1;
    }

    obj = PySequence_Fast(obj, CUSTOM_ERROR_NOT_A_SEQUENCE);

    npy_intp size = PySequence_Fast_GET_SIZE(obj);
    if (size < 0) {
        return -1;
    }

    PyObject **object = PySequence_Fast_ITEMS(obj);
    if (object == NULL) {
        return -1;
    }

    if (PyArray_UpdateShape(out_shape, size, curr_dim) < 0) {
        return -1;
    }

    int dim = curr_dim;

    for (npy_intp i = 0; i < size; i++) {
        dim = PyArray_DiscoverDTYpeAndShapeRecursive(object[i], max_dims, curr_dim + 1,
                                                     out_shape, out_descr);

        if (dim < 0)
            return -1;
    }

    return dim;
}

/*
 * @returns Returns the number of dimension of the PyObject that is passed in, while
 * populating the out shape to the corresponding shape of each dimension.
 * */
int
PyArray_DiscoverDTypeAndShape(PyObject *obj, int max_dims,
                              npy_intp out_shape[NPY_MAXDIMS],
                              PyArray_Descr **out_descr)
{
    // not a sequence, only a scalar value
    if (!PySequence_Check(obj)) {
        out_shape[0] = (npy_intp)1;
        return 1;
    }

    if (obj == NULL)
        return -1;

    // reset shape
    for (int i = 0; i < NPY_MAXDIMS; i++) {
        out_shape[i] = (npy_intp)0;
    }

    return PyArray_DiscoverDTYpeAndShapeRecursive(obj, max_dims, 1, out_shape,
                                                  out_descr);
}
