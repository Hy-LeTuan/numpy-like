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
    self->dimensions = malloc(sizeof(int) * ndims);
    for (int i = 0; i < ndims; i++) {
        self->dimensions[i] = (int)out_shape[i];
    }

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
    PyArray_HandleDataBuffer(self, obj);

    return 0;
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

static PyObject *
PyArrayObject_display_data(PyArrayObject *self, PyObject *Py_UNUSED(ignored))
{
    long long num_elem = 1;

    for (int i = 0; i < NPY_NDIM(self); i++) {
        num_elem *= NPY_DIM(self)[i];
    }

    int counter = 0;
    for (npy_intp i = 0; i < num_elem * NPY_ELSIZE(self); i += NPY_ELSIZE(self)) {
        char *ptr = malloc(NPY_ELSIZE(self));

        for (npy_intp j = 0; j < NPY_ELSIZE(self); j++) {
            ptr[j] = self->data[i + j];
        }

        float f = *((float *)ptr);
        printf("element at index: %d is: %f\n", counter, f);

        counter++;
    }

    Py_RETURN_NONE;
}

static PyObject *
PyArrayObject_getattr(PyArrayObject *obj, PyObject *attr)
{
    const char *name = PyUnicode_AsUTF8(attr);
    PyObject *out;

    if (name == NULL) {
        return NULL;
    }

    if (strcmp(name, "nd") == 0) {
        out = PyLong_FromLong((long)*obj->nd);
        return out;
    }
    else if (strcmp(name, "dimensions") == 0) {
        out = PyTuple_New((npy_intp)NPY_NDIM(obj));

        for (npy_intp i = 0; i < (npy_intp)NPY_NDIM(obj); i++) {
            PyTuple_SET_ITEM(out, i, PyLong_FromLong((long)NPY_DIM(obj)[i]));
        }

        return out;
    }

    return PyObject_GenericGetAttr((PyObject *)obj, attr);
}

static PyMethodDef PyArrayObject_methods[] = {
        {"display_shape", (PyCFunction)PyArrayObject_display_shape, METH_NOARGS,
         "Display the dimension (shape) of the array."},
        {"display_data", (PyCFunction)PyArrayObject_display_data, METH_NOARGS,
         "Display the data in the raw buffer of the array."},
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
        .tp_getattro = (getattrofunc)PyArrayObject_getattr,
        .tp_methods = PyArrayObject_methods,
};
