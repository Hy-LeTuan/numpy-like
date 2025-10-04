#include <Python.h>
#include <ndarray_types.h>

int
discover_shape_and_dtpye()
{
}

int
PyArray_DiscoverDTypeAndShape(PyObject *obj, int max_dims,
                              npy_intp out_shape[NPY_MAXDIMS], PyArray_Descr *out_descr)
{
}
