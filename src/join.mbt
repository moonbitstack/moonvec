// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
/// Joins arrays along an existing axis; the other axes must agree in length, and the dtypes promote.
pub fn concatenate(
  arrays : Array[NDArray],
  axis~ : Int32 = 0,
) -> NDArray raise VecError {
  if arrays.length() == 0 {
    raise ShapeMismatch(want=[0], got=[])
  }
  let first = arrays[0]
  let ndim = first.ndim()
  if ndim == 0 {
    raise ShapeMismatch(want=first.shape(), got=first.shape())
  }
  let position = axis_position(first, axis)
  let shapes : Array[Array[Int32]] = []
  let dtypes : Array[DType] = []
  let mut total = 0
  for array in arrays {
    if array.ndim() != ndim {
      raise ShapeMismatch(want=first.shape(), got=array.shape())
    }
    let shape = array.shape()
    for i in 0..<ndim {
      if i != position && shape[i] != first.shape()[i] {
        raise ShapeMismatch(want=first.shape(), got=shape)
      }
    }
    total = total + shape[position]
    shapes.push(shape)
    dtypes.push(array.dtype())
  }
  let out_shape = first.shape()
  out_shape[position] = total
  let out = alloc(out_shape, promote_all(dtypes[:]))
  let out_strides = contiguous_strides(out_shape)
  let mut placed = 0
  for i in 0..<arrays.length() {
    let shape = shapes[i]
    let offset = out.offset() + placed * out_strides[position]
    let target = NDArray::new(
      buffer=out.buffer,
      shape~,
      strides=out_strides,
      offset~,
    )
    copy_elements(target, arrays[i])
    placed = placed + shape[position]
  }
  out
}

///|
/// Joins arrays along a new axis, so the result has one more axis than the inputs.
pub fn stack(
  arrays : Array[NDArray],
  axis~ : Int32 = 0,
) -> NDArray raise VecError {
  if arrays.length() == 0 {
    raise ShapeMismatch(want=[0], got=[])
  }
  let ndim = arrays[0].ndim()
  let shape = arrays[0].shape()
  let lifted : Array[NDArray] = []
  for array in arrays {
    if array.shape() != shape {
      raise ShapeMismatch(want=shape, got=array.shape())
    }
    lifted.push(expand_dims(array, axis~))
  }
  let position = if axis < 0 { axis + ndim + 1 } else { axis }
  concatenate(lifted, axis=position)
}

///|
/// Splits an axis into equal parts; the length must divide evenly, as NumPy's `split` requires.
pub fn split(
  array : NDArray,
  parts~ : Int32,
  axis~ : Int32 = 0,
) -> Array[NDArray] raise VecError {
  if parts <= 0 {
    raise Domain(message="split needs a positive number of parts")
  }
  let position = axis_position(array, axis)
  let length = array.shape()[position]
  if length % parts != 0 {
    raise Domain(message="the axis does not divide into that many equal parts")
  }
  let step = length / parts
  let bounds : Array[Int32] = []
  for i in 1..<parts {
    bounds.push(i * step)
  }
  split_at(array, bounds, axis=position)
}

///|
/// Splits an axis at the given positions, which are the lengths of the parts before them. A position
/// outside the axis is clamped, which lets `split_at` stand in for `split`.
pub fn split_at(
  array : NDArray,
  at : Array[Int32],
  axis~ : Int32 = 0,
) -> Array[NDArray] raise VecError {
  let position = axis_position(array, axis)
  let length = array.shape()[position]
  let bounds : Array[Int32] = []
  let mut previous = 0
  for cut in at {
    let clamped = if cut < 0 { 0 } else if cut > length { length } else { cut }
    if clamped < previous {
      raise Domain(message="split positions must not go backwards")
    }
    bounds.push(clamped)
    previous = clamped
  }
  bounds.push(length)
  let parts : Array[NDArray] = []
  let mut start = 0
  for i in 0..<bounds.length() {
    let stop = bounds[i]
    let indices : Array[Index] = []
    for j in 0..<array.ndim() {
      if j == position {
        indices.push(Span(start, stop, 1))
      } else {
        indices.push(All)
      }
    }
    parts.push(slice(array, indices))
    start = stop
  }
  parts
}

///|
/// Rows one after another: a 1-D array becomes a single row, and anything with two or more axes is
/// joined along axis 0.
pub fn vstack(arrays : Array[NDArray]) -> NDArray raise VecError {
  let rows : Array[NDArray] = []
  for array in arrays {
    if array.ndim() == 0 {
      rows.push(reshape(array, [1, 1]))
    } else if array.ndim() == 1 {
      rows.push(reshape(array, [1, array.shape()[0]]))
    } else {
      rows.push(array)
    }
  }
  concatenate(rows, axis=0)
}

///|
/// Joins 1-D arrays end to end, and 2-D arrays column by column, which is the shape NumPy's `hstack`
/// has.
pub fn hstack(arrays : Array[NDArray]) -> NDArray raise VecError {
  if arrays.length() == 0 {
    raise ShapeMismatch(want=[0], got=[])
  }
  let ndim = arrays[0].ndim()
  if ndim == 1 {
    concatenate(arrays, axis=0)
  } else {
    concatenate(arrays, axis=1)
  }
}

///|
/// Copies the elements of `source` into `target`, which must have the same shape and enough room.
fn copy_elements(target : NDArray, source : NDArray) -> Unit raise VecError {
  if target.shape() != source.shape() {
    raise ShapeMismatch(want=target.shape(), got=source.shape())
  }
  let shape = target.shape()
  let plan = plan_loop(shape, [target.strides(), source.strides()], [
    target.offset(),
    source.offset(),
  ])
  let dtype = target.dtype()
  plan.run(fn(bases, len) raise VecError {
    let target_step = plan.run_strides[0]
    let source_step = plan.run_strides[1]
    for i in 0..<len {
      let value = source.buffer.read(bases[1] + i * source_step)
      target.buffer.write(
        bases[0] + i * target_step,
        Number::cast(value, dtype),
      )
    }
  })
}
