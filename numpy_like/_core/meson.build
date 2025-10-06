core_include = include_directories('include')

multiarray_sources = [
  'multiarray/multiarray.c',
  'multiarray/ndarray_types.c',
  'multiarray/array_coercion.c'
]

py.extension_module(
  'multiarray',
  sources: multiarray_sources,
  install: true,
  include_directories: core_include,
  subdir: 'numpy_like/_core'
)

python_sources = [
  '__init__.py',
]

py.install_sources(python_sources, subdir: 'numpy_like/_core')
