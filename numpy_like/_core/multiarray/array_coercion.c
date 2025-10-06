#include <Python.h>
#include <errors.h>
#include <ndarray_types.h>

/*
 * @returns Returns 1 if the function can populate out_descr and 0 otherwise.
 * */
int
PyArray_DeterminePyArrayDescr(PyObject *obj, PyArray_Descr *out_descr)
{
    return -1;
}

/**
 * @returns Returns -1 if shape mismatch happens
 * */
int
update_shape(npy_intp out_shape[NPY_MAXDIMS], int size, int curr_dim)
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
                                       PyArray_Descr *out_descr)
{
    if (!PySequence_Check(obj)) {
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

    if (update_shape(out_shape, size, curr_dim) < 0) {
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
                              npy_intp out_shape[NPY_MAXDIMS], PyArray_Descr *out_descr)
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
