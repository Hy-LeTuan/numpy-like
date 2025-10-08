#include <Python.h>
#include <array_coercion.h>
#include <ndarray_types.h>

#include "errors.h"

static void
PyArray_InitSetMetadata(PyArrayObject *self, PyObject *obj, const char *dtypes)
{
    free(self->strides);
    free(self->nd);
    free(self->dimensions);

    PyArray_Descr *descr = NULL;
    npy_intp ndims;
    npy_intp out_shape[NPY_MAXDIMS];

    ndims = PyArray_DiscoverDTypeAndShape(obj, NPY_MAXDIMS, out_shape, &descr);

    if (ndims < 0) {
        printf(CUSTOM_ERROR_DIMENSION_NOT_VALID);
    }
    else if (ndims > NPY_MAXDIMS) {
        printf(CUSTOM_ERROR_DIMENSION_NOT_VALID);
    }

    // set number of dimension
    self->nd = malloc(sizeof(int));
    *self->nd = (int)ndims;

    // set shape
    self->dimensions = (int *)out_shape;

    // set description
    self->descr = descr;

    // set strides, naively assuming no sparse array
    self->strides = malloc(sizeof(int) * ndims);
    for (int i = 0; i < ndims; i++) {
        self->strides[i] = self->descr->elsize;
    }
}

static int
PyArrayObject_init(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *obj;

    static char *keywords[] = {"array", "dtypes", NULL};
    const char *dtypes = "";

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|s", keywords, &obj, &dtypes)) {
        return -1;
    }

    // set metadata
    PyArray_InitSetMetadata(self, obj, dtypes);

    // read buffer (read buffer is only called) after metadata is discovered
    PyArray_ReadAndReturnRawBuffer(self, obj);

    return 0;
}

static PyObject *
PyArrayObject_display(PyArrayObject *self, PyObject *Py_UNUSED(ignored))
{
    int strides = self->strides[0];
    int dim = self->dimensions[0];

    for (int i = 0; i < strides * dim; i += strides) {
        char *num = malloc(strides);

        for (int j = 0; j < strides; j++) {
            num[j] = self->data[i + j];
        }

        // only the final cast is needed
        double *x = (double *)num;
        printf("the number x is: %f\n", *x);
    }

    Py_RETURN_NONE;
}

static PyObject *
PyArrayObject_display_shape(PyArrayObject *self, PyObject *Py_UNUSED(ignored))
{
    int nd = (int)*self->nd;

    for (int i = 0; i < nd; ++i) {
        printf("dimension %d has size: %d\n", i, (int)self->dimensions[i]);
    }

    Py_RETURN_NONE;
}

static PyMethodDef PyArrayObject_methods[] = {
        {"display", (PyCFunction)PyArrayObject_display, METH_NOARGS,
         "Display the array stored in the raw data pointer"},
        {"display_shape", (PyCFunction)PyArrayObject_display_shape, METH_NOARGS,
         "Display the dimension (shape) of the array."},
        {NULL}};

PyTypeObject PyArrayObjectType = {
        .ob_base = PyVarObject_HEAD_INIT(NULL, 0).tp_name = "numpy_like.ndarray",
        .tp_basicsize = sizeof(PyArrayObject),
        .tp_itemsize = 0,
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_doc = PyDoc_STR("ndarray objects"),

        // methods
        .tp_new = PyType_GenericNew,
        .tp_init = (initproc)PyArrayObject_init,
        .tp_methods = PyArrayObject_methods,
};
