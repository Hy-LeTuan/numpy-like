#include <Python.h>
#include <ndarray_types.h>

static int PyArrayObject_init(PyArrayObject *self, PyObject *args,
                              PyObject *kwds) {
    PyObject *array;
    int array_length;
    double *ptr;

    if (!PyArg_ParseTuple(args, "O", &array))
        return -1;

    array_length = PyObject_Length(array);

    if (array_length < 0)
        return -1;

    ptr = (double *)malloc(sizeof(double) * array_length);

    if (ptr == NULL)
        return -1;

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

    free(self->nd);
    self->nd = (int *)malloc(sizeof(int));
    *(self->nd) = 1;

    free(self->dimensions);
    self->dimensions = (int *)malloc(sizeof(int));
    *(self->dimensions) = array_length;

    free(self->strides);
    self->strides = (int *)malloc(sizeof(int));
    *(self->strides) = 1;

    return 0;
}

static PyObject *PyArrayObject_display(PyArrayObject *self,
                                       PyObject *Py_UNUSED(ignored)) {
    double *x = ((double *)self->data);
    int strides = self->strides[0];
    int dim = self->dimensions[0];

    for (int i = 0; i < dim; i += strides) {
        printf("number is: %f\n", x[i]);
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
