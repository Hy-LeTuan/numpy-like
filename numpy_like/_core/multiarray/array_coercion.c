#include "array_coercion.h"

#include <Python.h>
#include <errors.h>
#include <ndarray_types.h>

#include "abstract.h"
#include "boolobject.h"

PyArray_Descr *
PyArray_CreatePyArrayDescr(enum NPY_TYPES type)
{
    PyArray_Descr *out = malloc(sizeof(PyArray_Descr));

    switch (type) {
        case NPY_BOOL:
            out->dtype_obj = NULL;
            out->type = '?';
            out->type_num = (int)NPY_BOOL;
            out->byteorder = '>';
            out->elsize = 1;
            break;
        case NPY_BYTE:
            out->dtype_obj = NULL;
            out->type = 'b';
            out->type_num = (int)NPY_BYTE;
            out->byteorder = '>';
            out->elsize = 1;
            break;
        case NPY_UBYTE:
            out->dtype_obj = NULL;
            out->type = 'B';
            out->type_num = (int)NPY_UBYTE;
            out->byteorder = '>';
            out->elsize = 1;
            break;
        case NPY_SHORT:
            out->dtype_obj = NULL;
            out->type = 'h';
            out->type_num = (int)NPY_SHORT;
            out->byteorder = '>';
            out->elsize = 2;
            break;
        case NPY_USHORT:
            out->dtype_obj = NULL;
            out->type = 'H';
            out->type_num = (int)NPY_USHORT;
            out->byteorder = '>';
            out->elsize = 2;
            break;
        case NPY_INT:
            out->dtype_obj = NULL;
            out->type = 'i';
            out->type_num = (int)NPY_USHORT;
            out->byteorder = '>';
            out->elsize = 4;
            break;
        case NPY_UINT:
            out->dtype_obj = NULL;
            out->type = 'I';
            out->type_num = (int)NPY_UINT;
            out->byteorder = '>';
            out->elsize = 4;
            break;
        case NPY_LONG:
            out->dtype_obj = NULL;
            out->type = 'l';
            out->type_num = (int)NPY_LONG;
            out->byteorder = '>';
            out->elsize = 8;
            break;
        case NPY_ULONG:
            out->dtype_obj = NULL;
            out->type = 'L';
            out->type_num = (int)NPY_ULONG;
            out->byteorder = '>';
            out->elsize = 8;
            break;
        case NPY_LONGLONG:
            out->dtype_obj = NULL;
            out->type = 'q';
            out->type_num = (int)NPY_LONGLONG;
            out->byteorder = '>';
            out->elsize = 8;
            break;
        case NPY_ULONGLONG:
            out->dtype_obj = NULL;
            out->type = 'Q';
            out->type_num = (int)NPY_ULONGLONG;
            out->byteorder = '>';
            out->elsize = 8;
            break;
        case NPY_FLOAT:
            out->dtype_obj = NULL;
            out->type = 'f';
            out->type_num = (int)NPY_FLOAT;
            out->byteorder = '>';
            out->elsize = 4;
            break;
        case NPY_DOUBLE:
            out->dtype_obj = NULL;
            out->type = 'd';
            out->type_num = (int)NPY_DOUBLE;
            out->byteorder = '>';
            out->elsize = 8;
            break;
        case NPY_LONGDOUBLE:
            out->dtype_obj = NULL;
            out->type = 'g';
            out->type_num = (int)NPY_LONGDOUBLE;
            out->byteorder = '>';
            out->elsize = 16;
            break;
        case NPY_CFLOAT:
            out->dtype_obj = NULL;
            out->type = 'F';
            out->type_num = (int)NPY_CFLOAT;
            out->byteorder = '>';
            out->elsize = 4;
            break;
        case NPY_CDOUBLE:
            out->dtype_obj = NULL;
            out->type = 'D';
            out->type_num = (int)NPY_CDOUBLE;
            out->byteorder = '>';
            out->elsize = 8;
            break;
        case NPY_CLONGDOUBLE:
            out->dtype_obj = NULL;
            out->type = 'G';
            out->type_num = (int)NPY_CLONGDOUBLE;
            out->byteorder = '>';
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

/**
 * Support function to assign a char* item data storage to an array buffer. The type of
 * `item` is already known and handled by the caller.
 * **/
void
PyArray_AssigElementToBufferChar(PyArrayObject *self, char *converted, void *buffer,
                                 int index)
{
    for (npy_intp j = 0; j < NPY_ELSIZE(self); j++) {
        ((char *)buffer)[index * NPY_ELSIZE(self) + j] = converted[j];
    }
}

/**
 * The function to parse the raw PyObject scalar value into the approrpiate type, then
 * use type conversion to assign it back into the buffer.
 * @param index: The index of the start of the element inside the flat buffer
 * **/
void
PyArray_ParseAndAssignElementToBuffer(PyArrayObject *self, PyObject *item, void *buffer,
                                      int index)
{
    char *converted;

    if (PyLong_Check(item)) {
        long parsed = PyLong_AsLong(item);
        long *temp = &parsed;
        converted = (char *)temp;

        PyArray_AssigElementToBufferChar(self, converted, buffer, index);
    }
    else if (PyFloat_Check(item)) {
        float parsed = PyFloat_AsDouble(item);
        float *temp = &parsed;
        converted = (char *)temp;

        PyArray_AssigElementToBufferChar(self, converted, buffer, index);
    }
}

/**
 * This function computes the flattened array index (index into the raw data buffer) for
 * any given multi-dimensional index through the use of the ndarray's metadata.
 * @returns Returns the flattened index from the dimensional index and the dimensions of
 * the current ndarray if succeed, 0 otherwise.
 * **/
int
PyArray_ComputeFlattenedIndex(int *index, int *dimensions, int nd)
{
    int flattened_index = 0;
    int current_dim = 1;

    for (int i = nd - 1; i >= 0; i--) {
        flattened_index += index[i] * current_dim;
        current_dim *= dimensions[i];
    }

    return flattened_index;
}

/**
 * @returns Returns -1 if errors occured when writing data to buffer, and 0 otherwise.
 * **/
int
PyArray_ReadAndReturnRawBuffferRecursive(void *buffer, PyArrayObject *self,
                                         PyObject *obj, int *index, int level)
{
    PyObject *object;
    object = PySequence_Fast(obj, CUSTOM_ERROR_NOT_A_SEQUENCE);

    if (level == NPY_NDIM(self)) {
        for (int i = 0; i < NPY_DIM(self)[level - 1]; i++) {
            PyObject *item;
            item = PySequence_Fast_GET_ITEM(object, i);

            index[level - 1] = i;
            int flattened_index =
                    PyArray_ComputeFlattenedIndex(index, NPY_DIM(self), NPY_NDIM(self));

            PyArray_ParseAndAssignElementToBuffer(self, item, buffer, flattened_index);
        }
    }
    else {
        for (int i = 0; i < self->dimensions[level - 1]; i++) {
            index[level - 1] = i;
            PyObject *item;
            item = PySequence_Fast_GET_ITEM(object, i);

            PyArray_ReadAndReturnRawBuffferRecursive(buffer, self, item, index,
                                                     level + 1);
        }
    }

    return 0;
}

void *
PyArray_ReadAndReturnRawBuffer(PyArrayObject *self, PyObject *obj)
{
    int ndims = NPY_NDIM(self);
    long long num_elem = 1;

    void *buffer;
    int *index;

    for (int i = 0; i < ndims; i++) {
        num_elem *= NPY_DIM(self)[i];
    }

    buffer = malloc(NPY_ELSIZE(self) * num_elem);
    index = malloc(sizeof(int) * ndims);

    PyArray_ReadAndReturnRawBuffferRecursive(buffer, self, obj, index, 1);

    return buffer;
}

int
PyArray_HandleDataBuffer(PyArrayObject *self, PyObject *obj)
{
    if (!PySequence_Check(obj)) {
        printf(CUSTOM_ERROR_NOT_A_SEQUENCE);
        return -1;
    }

    if (self->data != NULL) {
        printf(CUSTOM_WARNING_OVERRIDING_CURRENT_DATA);
    }

    free(self->data);

    void *buff = PyArray_ReadAndReturnRawBuffer(self, obj);

    self->data = (char *)buff;

    return 0;
}
