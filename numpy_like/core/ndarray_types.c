#include <Python.h>
#include <ndarray_types.h>
#include <string.h>

static void init_set_meta_data(PyArrayObject *self, const char *dtypes,
                               int array_length) {
    free(self->strides);
    free(self->nd);
    free(self->dimensions);

    if (strcmp(dtypes, "float32") == 0) {
        self->strides = malloc(sizeof(int));
        *self->strides = 4;
    } else if (!*dtypes || strcmp(dtypes, "float64") == 0) {
        self->strides = malloc(sizeof(int));
        *self->strides = 8;
    }

    self->nd = (int *)malloc(sizeof(int));
    *(self->nd) = 1;

    self->dimensions = (int *)malloc(sizeof(int));
    *(self->dimensions) = array_length;
}

static int PyArrayObject_init(PyArrayObject *self, PyObject *args,
                              PyObject *kwds) {
    PyObject *array;
    int array_length;
    double *ptr;

    static char *keywords[] = {"array", "dtypes", NULL};
    const char *dtypes = "";

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|s", keywords, &array,
                                     &dtypes)) {
        return -1;
    }

    array_length = PyObject_Length(array);
    if (array_length < 0) {
        return -1;
    }

    ptr = (double *)malloc(sizeof(double) * array_length);

    if (ptr == NULL)
        return -1;

    // populate buffer data
    for (int i = 0; i < array_length; i++) {
        PyObject *item;
        item = PyList_GetItem(array, i);

        if (!PyFloat_Check(item)) {
            ptr[i] = 0.0;
        } else {
            ptr[i] = PyFloat_AsDouble(item);
        }
    }

    free(self->data);
    self->data = (char *)ptr;

    // set metadata
    init_set_meta_data(self, dtypes, array_length);

    return 0;
}

static PyObject *PyArrayObject_display(PyArrayObject *self,
                                       PyObject *Py_UNUSED(ignored)) {
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

static PyMethodDef PyArrayObject_methods[] = {
    {"display", (PyCFunction)PyArrayObject_display, METH_NOARGS,
     "Display the array stored in the raw data pointer"},
    {NULL}};

PyTypeObject PyArrayObjectType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0).tp_name = "numpy_like.ndarray",
    .tp_basicsize = sizeof(PyArrayObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = PyDoc_STR("ndarray objects"),
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)PyArrayObject_init,
    .tp_methods = PyArrayObject_methods,
};
