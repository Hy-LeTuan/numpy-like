#include <Python.h>
#include <ndarray_types.h>

static int
core_module_exec(PyObject *m)
{
    if (PyType_Ready(&(PyArrayObjectType)) < 0) {
        return -1;
    }

    if (PyModule_AddObjectRef(m, "ndarray", (PyObject *)&PyArrayObjectType) < 0) {
        return -1;
    }

    return 0;
}

static PyModuleDef_Slot module_slots[] = {
        {Py_mod_exec, core_module_exec},
        {Py_mod_multiple_interpreters, Py_MOD_MULTIPLE_INTERPRETERS_SUPPORTED},
        {0, NULL},
};

static PyModuleDef module = {
        .m_base = PyModuleDef_HEAD_INIT,
        .m_name = "numpy_like._core.multiarray",
        .m_doc = "The multiarray module, a module that handles array creation, which "
                 "is a part of the _core namespace",
        .m_slots = module_slots,
        .m_size = 0,
};

PyMODINIT_FUNC
PyInit_multiarray(void)
{
    return PyModuleDef_Init(&module);
}
