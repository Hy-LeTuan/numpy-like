project('purelib-and-platlib', 'c')

py = import('python').find_installation(pure: false)

py.extension_module(
  'numpy_like',
  './src/numpy_like.c',
  install: true,
)
