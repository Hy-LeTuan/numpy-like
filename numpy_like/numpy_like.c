#include <Python.h>
#include <ndarray_types.h>

static int numpy_like_module_exec(PyObject *m) {
    if (PyType_Ready(&(PyArrayObjectType)) < 0) {
        return -1;
    }

    if (PyModule_AddObjectRef(m, "ndarray", (PyObject *)&PyArrayObjectType) <
        0) {
        return -1;
    }

    return 0;
}

static PyModuleDef_Slot module_slots[] = {
    {Py_mod_exec, numpy_like_module_exec},
    {Py_mod_multiple_interpreters, Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED},
    {0, NULL},
};

static PyModuleDef module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "numpy_like",
    .m_doc = "Numpy clone",
    .m_size = 0,
    .m_slots = module_slots,
};

PyMODINIT_FUNC PyInit_numpy_like(void) { return PyModuleDef_Init(&module); }
