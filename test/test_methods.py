import numpy_like

a = numpy_like.ndarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtypes="float32")

print(f"Dimension is: {a.dimensions}")
assert a.nd == 2, "Test failed, the number of dimension of a 2D array should be 2."

a.display_data()
