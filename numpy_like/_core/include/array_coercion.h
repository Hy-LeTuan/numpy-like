#ifndef CORE_ARRAY_COERCION_H_
#define CORE_ARRAY_COERCION_H_

#include <Python.h>
#include <ndarray_types.h>

PyArray_Descr *
PyArray_CreatePyArrayDescr(enum NPY_TYPES type);

int
PyArray_DeterminePyArrayDescr(PyObject *obj, PyArray_Descr **out_descr);

int
PyArray_DiscoverDTYpeAndShapeRecursive(PyObject *obj, int max_dims, int curr_dim,
                                       npy_intp out_shape[NPY_MAXDIMS],
                                       PyArray_Descr **out_descr);

int
PyArray_DiscoverDTypeAndShape(PyObject *obj, int max_dims,
                              npy_intp out_shape[NPY_MAXDIMS],
                              PyArray_Descr **out_descr);

int
PyArray_ComputeFlattenedIndex(int *index, int *dimensions, int nd);

void
PyArray_ParseAndAssignElementToBuffer(PyArrayObject *self, PyObject *item, void *buffer,
                                      int index);

int
PyArray_HandleDataBuffer(PyArrayObject *self, PyObject *obj);

void *
PyArray_ReadAndReturnRawBuffer(PyArrayObject *self, PyObject *obj);

#endif
