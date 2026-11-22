// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
/// One axis selector in a slice, read left to right, one per axis.
pub(all) enum Index {
  ///
  /// The whole axis.
  All
  ///
  /// A single position, which removes the axis from the result. A negative value counts from the
  /// end.
  At(Int32)
  ///
  /// A run of positions: start, stop (exclusive) and step, with negative indices counting from the
  /// end and a negative step walking backwards. Out-of-range bounds are clamped rather than refused.
  Span(Int32, Int32, Int32)
  ///
  /// A new axis of length 1.
  NewAxis
  ///
  /// Integer array indexing along this axis; the result is a copy.
  Take(Array[Int32])
  ///
  /// A boolean mask along this axis; the result is a copy.
  Mask(Array[Bool])
}

///|
/// A run of positions, the common case written without spelling out the step.
pub fn span(start : Int32, stop : Int32, step? : Int32) -> Index {
  Span(
    start,
    stop,
    match step {
      Some(step) => step
      None => 1
    },
  )
}

///|
/// A view of `array` selected axis by axis; axes past the end of `indices` are taken whole.
///
/// Slicing a contiguous array stays a view: nothing is copied, so a write through the slice is a
/// write into the original. `At` removes an axis and `NewAxis` adds one. `Take` and `Mask` cannot be
/// expressed by strides, so they copy.
pub fn slice(array : NDArray, indices : Array[Index]) -> NDArray raise VecError {
  let mut consumed = 0
  for index in indices {
    if index is NewAxis {
      continue
    }
    consumed = consumed + 1
  }
  if consumed > array.ndim() {
    raise ShapeMismatch(want=array.shape(), got=[indices.length()])
  }
  let shape : Array[Int32] = []
  let strides : Array[Int32] = []
  let gathers : Array[(Int32, Index)] = []
  let mut offset = array.offset()
  let source_shape = array.shape()
  let source_strides = array.strides()
  let mut source_axis = 0
  for i in 0..<indices.length() {
    match indices[i] {
      NewAxis => {
        shape.push(1)
        strides.push(0)
      }
      All => {
        shape.push(source_shape[source_axis])
        strides.push(source_strides[source_axis])
        source_axis = source_axis + 1
      }
      At(position) => {
        let picked = normalize_index(position, source_shape[source_axis])
        offset = offset + picked * source_strides[source_axis]
        source_axis = source_axis + 1
      }
      Span(start, stop, step) => {
        let (begin, length, walk) = span_bounds(
          start,
          stop,
          step,
          source_shape[source_axis],
        )
        offset = offset + begin * source_strides[source_axis]
        shape.push(length)
        strides.push(source_strides[source_axis] * walk)
        source_axis = source_axis + 1
      }
      Take(_) | Mask(_) => {
        shape.push(source_shape[source_axis])
        strides.push(source_strides[source_axis])
        gathers.push((shape.length() - 1, indices[i]))
        source_axis = source_axis + 1
      }
    }
  }
  for i in source_axis..<array.ndim() {
    shape.push(source_shape[i])
    strides.push(source_strides[i])
  }
  let view = NDArray::new(buffer=array.buffer, shape~, strides~, offset~)
  let mut result = view
  for gather in gathers {
    let (axis, selector) = gather
    result = match selector {
      Take(picks) => take(result, picks, axis~)
      Mask(keep) => mask(result, keep, axis~)
      _ => result
    }
  }
  result
}

///|
/// A position on an axis, negative counting from the end.
fn normalize_index(position : Int32, length : Int32) -> Int32 raise VecError {
  let at = if position < 0 { position + length } else { position }
  if at < 0 || at >= length {
    raise IndexOutOfBounds(index=position, len=length)
  }
  at
}

///|
/// Clamps a start, stop and step into the axis, and reports how many positions the run has.
fn span_bounds(
  start : Int32,
  stop : Int32,
  step : Int32,
  length : Int32,
) -> (Int32, Int32, Int32) {
  if step == 0 {
    return (0, 0, 1)
  }
  let mut begin = if start < 0 { start + length } else { start }
  let mut end = if stop < 0 { stop + length } else { stop }
  if step > 0 {
    if begin < 0 {
      begin = 0
    }
    if begin > length {
      begin = length
    }
    if end < 0 {
      end = 0
    }
    if end > length {
      end = length
    }
    let count = if end > begin { (end - begin + step - 1) / step } else { 0 }
    (begin, count, step)
  } else {
    if begin < 0 {
      begin = -1
    }
    if begin >= length {
      begin = length - 1
    }
    if end < -1 {
      end = -1
    }
    if end >= length {
      end = length - 1
    }
    let count = if begin > end { (begin - end - step - 1) / -step } else { 0 }
    (begin, count, step)
  }
}

///|
/// A view with the same elements in a new shape; `-1` in one position asks for the size that fits.
///
/// A contiguous array reshapes as a view. A strided one is copied first, because there is no stride
/// pattern that would make the new shape a view of the old.
pub fn reshape(array : NDArray, shape : Array[Int32]) -> NDArray raise VecError {
  let size = array.size()
  let target = fill_inferred(shape, size)
  let total = element_count(target)
  if total != size {
    raise CannotReshape(size~, shape=target)
  }
  let source = if array.is_contiguous() { array } else { array.copy() }
  NDArray::new(buffer=source.buffer, shape=target, offset=source.offset())
}

///|
/// Replaces one `-1` with the size that makes the product match.
fn fill_inferred(
  shape : Array[Int32],
  size : Int32,
) -> Array[Int32] raise VecError {
  let out = shape.copy()
  let mut missing = -1
  let mut known = 1
  for i in 0..<out.length() {
    if out[i] < 0 {
      if missing >= 0 {
        raise CannotReshape(size~, shape~)
      }
      missing = i
    } else {
      known = known * out[i]
    }
  }
  if missing >= 0 {
    if known == 0 || size % known != 0 {
      raise CannotReshape(size~, shape~)
    }
    out[missing] = size / known
  }
  out
}

///|
/// A 1-D view when the array is contiguous, a 1-D copy when it is not.
pub fn ravel(array : NDArray) -> NDArray raise VecError {
  if array.is_contiguous() {
    NDArray::new(
      buffer=array.buffer,
      shape=[array.size()],
      offset=array.offset(),
    )
  } else {
    let copied = array.copy()
    NDArray::new(
      buffer=copied.buffer,
      shape=[array.size()],
      offset=copied.offset(),
    )
  }
}

///|
/// A 1-D copy, always.
pub fn flatten(array : NDArray) -> NDArray raise VecError {
  let copied = array.copy()
  NDArray::new(
    buffer=copied.buffer,
    shape=[array.size()],
    offset=copied.offset(),
  )
}

///|
/// A view with the axes permuted; without `axes` the order is reversed.
pub fn transpose(
  array : NDArray,
  axes? : Array[Int32],
) -> NDArray raise VecError {
  let ndim = array.ndim()
  let order = match axes {
    Some(axes) => {
      if axes.length() != ndim {
        raise ShapeMismatch(want=array.shape(), got=axes)
      }
      let seen = Array::make(ndim, false)
      for axis in axes {
        let position = if axis < 0 { axis + ndim } else { axis }
        if position < 0 || position >= ndim || seen[position] {
          raise AxisOutOfRange(axis~, ndim~)
        }
        seen[position] = true
      }
      axes.map(fn(axis) { if axis < 0 { axis + ndim } else { axis } })
    }
    None => {
      let order = Array::make(ndim, 0)
      for i in 0..<ndim {
        order[i] = ndim - 1 - i
      }
      order
    }
  }
  let shape = Array::make(ndim, 0)
  let strides = Array::make(ndim, 0)
  let source_shape = array.shape()
  let source_strides = array.strides()
  for i in 0..<ndim {
    shape[i] = source_shape[order[i]]
    strides[i] = source_strides[order[i]]
  }
  NDArray::new(buffer=array.buffer, shape~, strides~, offset=array.offset())
}

///|
/// A view with two axes exchanged.
pub fn swap_axes(
  array : NDArray,
  first : Int32,
  second : Int32,
) -> NDArray raise VecError {
  let ndim = array.ndim()
  let a = if first < 0 { first + ndim } else { first }
  let b = if second < 0 { second + ndim } else { second }
  if a < 0 || a >= ndim {
    raise AxisOutOfRange(axis=first, ndim~)
  }
  if b < 0 || b >= ndim {
    raise AxisOutOfRange(axis=second, ndim~)
  }
  let order = Array::make(ndim, 0)
  for i in 0..<ndim {
    order[i] = i
  }
  order[a] = b
  order[b] = a
  transpose(array, axes=order)
}

///|
/// A view with one axis moved to another position, the other axes keeping their order.
pub fn move_axis(
  array : NDArray,
  from : Int32,
  to : Int32,
) -> NDArray raise VecError {
  let ndim = array.ndim()
  let source = if from < 0 { from + ndim } else { from }
  let target = if to < 0 { to + ndim } else { to }
  if source < 0 || source >= ndim {
    raise AxisOutOfRange(axis=from, ndim~)
  }
  if target < 0 || target >= ndim {
    raise AxisOutOfRange(axis=to, ndim~)
  }
  let order : Array[Int32] = []
  for i in 0..<ndim {
    if i != source {
      order.push(i)
    }
  }
  order.insert(target, source)
  transpose(array, axes=order)
}

///|
/// A view with axes of length 1 removed; with `axis`, only that axis and only if it has length 1.
pub fn squeeze(array : NDArray, axis? : Int32) -> NDArray raise VecError {
  let shape : Array[Int32] = []
  let strides : Array[Int32] = []
  let source_shape = array.shape()
  let source_strides = array.strides()
  match axis {
    Some(axis) => {
      let position = axis_position(array, axis)
      if source_shape[position] != 1 {
        raise ShapeMismatch(want=[1], got=[source_shape[position]])
      }
      for i in 0..<array.ndim() {
        if i != position {
          shape.push(source_shape[i])
          strides.push(source_strides[i])
        }
      }
    }
    None =>
      for i in 0..<array.ndim() {
        if source_shape[i] != 1 {
          shape.push(source_shape[i])
          strides.push(source_strides[i])
        }
      }
  }
  NDArray::new(buffer=array.buffer, shape~, strides~, offset=array.offset())
}

///|
/// A view with a new axis of length 1 at `axis`.
pub fn expand_dims(array : NDArray, axis~ : Int32) -> NDArray raise VecError {
  let ndim = array.ndim()
  let position = if axis < 0 { axis + ndim + 1 } else { axis }
  if position < 0 || position > ndim {
    raise AxisOutOfRange(axis~, ndim=ndim + 1)
  }
  let shape = array.shape()
  let strides = array.strides()
  shape.insert(position, 1)
  strides.insert(position, 0)
  NDArray::new(buffer=array.buffer, shape~, strides~, offset=array.offset())
}

///|
/// The position of `axis`, negative counting from the end.
fn axis_position(array : NDArray, axis : Int32) -> Int32 raise VecError {
  let ndim = array.ndim()
  let position = if axis < 0 { axis + ndim } else { axis }
  if position < 0 || position >= ndim {
    raise AxisOutOfRange(axis~, ndim~)
  }
  position
}

///|
/// Picks positions along one axis by an integer array, keeping the number of axes; the result is a
/// copy.
pub fn take(
  array : NDArray,
  picks : Array[Int32],
  axis? : Int32,
) -> NDArray raise VecError {
  match axis {
    None => {
      let flat = ravel(array)
      take(flat, picks, axis=0)
    }
    Some(axis) => {
      let position = axis_position(array, axis)
      let length = array.shape()[position]
      let resolved = picks.map(fn(pick) raise VecError {
        normalize_index(pick, length)
      })
      let shape = array.shape()
      shape[position] = picks.length()
      gather(array, shape, position, resolved)
    }
  }
}

///|
/// Keeps the positions where the mask is true along one axis; the mask must have the axis's length.
pub fn mask(
  array : NDArray,
  keep : Array[Bool],
  axis? : Int32,
) -> NDArray raise VecError {
  let position = match axis {
    Some(axis) => axis_position(array, axis)
    None => 0
  }
  if keep.length() != array.shape()[position] {
    raise ShapeMismatch(want=array.shape(), got=[keep.length()])
  }
  let picks : Array[Int32] = []
  for i in 0..<keep.length() {
    if keep[i] {
      picks.push(i)
    }
  }
  take(array, picks, axis=position)
}

///|
/// Copies the elements at `picks` along `position` into a new contiguous array of `shape`.
fn gather(
  array : NDArray,
  shape : Array[Int32],
  position : Int32,
  picks : Array[Int32],
) -> NDArray raise VecError {
  let out = alloc(shape, array.dtype())
  let ndim = shape.length()
  let strides = array.strides()
  let index = Array::make(ndim, 0)
  let count = element_count(shape)
  for i in 0..<count {
    let mut offset = array.offset()
    for axis in 0..<ndim {
      let coordinate = if axis == position {
        picks[index[axis]]
      } else {
        index[axis]
      }
      offset = offset + coordinate * strides[axis]
    }
    out.buffer.write(i, array.buffer.read(offset))
    let mut axis = ndim - 1
    while axis >= 0 {
      index[axis] = index[axis] + 1
      if index[axis] < shape[axis] {
        break
      }
      index[axis] = 0
      axis = axis - 1
    }
  }
  out
}
