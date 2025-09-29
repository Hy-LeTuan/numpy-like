project('purelib-and-platlib', 'c')

incdir = include_directories('./src/core/include')

py = import('python').find_installation(pure: false)

py.extension_module(
  'numpy_like',
  './src/numpy_like.c',
  install: true,
  include_directories: incdir
)
