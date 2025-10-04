project('purelib-and-platlib', 'c')

incdir = include_directories('./numpy_like/core/include')

py = import('python').find_installation(pure: false)

py.extension_module(
  'numpy_like',
  './numpy_like/numpy_like.c',
  './numpy_like/core/ndarray_types.c',
  './numpy_like/core/array_coercion.c',
  install: true,
  include_directories: incdir
)
