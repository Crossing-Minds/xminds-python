import numpy


def structured_cast(array, dtype):
    """
    if array is a structured array, align fields by names and cast to dtype
    if any field with name '__pad__' is present in `dtype` but not in `array`,
    then output will include padding zeros
    :param array array:
    :param dtype-descr dtype:
    :returns: array
    """
    dtype = numpy.dtype(dtype)
    if array.dtype == dtype:
        return array
    if not dtype.names:
        return array.astype(dtype)
    if not('__pad__' in dtype.names and '__pad__' not in array.dtype.names):
        return array[list(dtype.names)].astype(dtype)

    new_array = numpy.empty_like(array, dtype=dtype)
    for field_name, (field_dtype, _) in dtype.fields.items():
        if field_name == '__pad__':
            new_array[field_name] = numpy.zeros(1, field_dtype)[0]
        else:
            new_array[field_name] = array[field_name].astype(field_dtype)
    return new_array
