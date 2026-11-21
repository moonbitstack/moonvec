// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
/// An N-dimensional array: a view over a flat buffer, described by a shape and a stride per axis.
///
/// The strides are counted in elements, not bytes, and follow the C order NumPy uses: the last axis
/// varies fastest. Any axis may have stride 0, which is how a size-one axis broadcasts without
/// copying its data, and any axis may be negative, which is how a reversed view is made. Two arrays
/// made by slicing the same array share one buffer, so a write through one is visible through the
/// other, as with NumPy's views.
pub struct NDArray {
  buffer : Buffer
  shape : Array[Int32]
  strides : Array[Int32]
  offset : Int32
}

///|
/// Builds a view over `buffer`; the strides default to the contiguous layout of `shape`.
pub fn NDArray::new(
  buffer~ : Buffer,
  shape~ : Array[Int32],
  strides? : Array[Int32],
  offset~ : Int32 = 0,
) -> NDArray raise VecError {
  for dim in shape {
    if dim < 0 {
      raise CannotReshape(size=buffer.len(), shape~)
    }
  }
  let strides = match strides {
    Some(strides) => {
      if strides.length() != shape.length() {
        raise ShapeMismatch(want=shape, got=strides)
      }
      strides
    }
    None => contiguous_strides(shape)
  }
  if offset < 0 {
    raise IndexOutOfBounds(index=offset, len=buffer.len())
  }
  let mut reach = offset
  for i in 0..<shape.length() {
    if shape[i] > 1 {
      let step = strides[i] * (shape[i] - 1)
      if step > 0 {
        reach = reach + step
      }
    }
  }
  if shape.length() > 0 && element_count(shape) > 0 && reach >= buffer.len() {
    raise IndexOutOfBounds(index=reach, len=buffer.len())
  }
  { buffer, shape, strides, offset, }
}

///|
pub fn NDArray::buffer(self : NDArray) -> Buffer {
  self.buffer
}

///|
/// The size of each axis; a copy, so changing it cannot break the view.
pub fn NDArray::shape(self : NDArray) -> Array[Int32] {
  self.shape.copy()
}

///|
/// The element distance between neighbours along each axis; a copy.
pub fn NDArray::strides(self : NDArray) -> Array[Int32] {
  self.strides.copy()
}

///|
/// The index in the buffer of the first element, after any slicing.
pub fn NDArray::offset(self : NDArray) -> Int32 {
  self.offset
}

///|
pub fn NDArray::ndim(self : NDArray) -> Int32 {
  self.shape.length()
}

///|
pub fn NDArray::size(self : NDArray) -> Int32 {
  element_count(self.shape)
}

///|
pub fn NDArray::dtype(self : NDArray) -> DType {
  self.buffer.dtype
}

///|
/// True for a 0-dimensional array, the result of a full reduction.
pub fn NDArray::is_scalar(self : NDArray) -> Bool {
  self.shape.length() == 0
}

///|
/// True when the elements sit next to each other in buffer order, so a backend can take the whole
/// array as one run.
pub fn NDArray::is_contiguous(self : NDArray) -> Bool {
  if self.shape.length() == 0 {
    return true
  }
  let mut expected = 1
  let mut i = self.shape.length() - 1
  while i >= 0 {
    if self.shape[i] != 1 && self.strides[i] != expected {
      return false
    }
    expected = expected * self.shape[i]
    i = i - 1
  }
  true
}

///|
/// The offset of the element at `index`, without checking that `index` is inside the shape.
fn NDArray::offset_of(self : NDArray, index : ArrayView[Int32]) -> Int32 {
  let mut at = self.offset
  for i in 0..<index.length() {
    at = at + index[i] * self.strides[i]
  }
  at
}

///|
/// The element at `index`, one integer per axis.
pub fn NDArray::at(
  self : NDArray,
  index : Array[Int32],
) -> Number raise VecError {
  check_index(self.shape, index)
  self.buffer.read(self.offset_of(index))
}

///|
/// Writes one element at `index`.
pub fn NDArray::set_at(
  self : NDArray,
  index : Array[Int32],
  value : Number,
) -> Unit raise VecError {
  check_index(self.shape, index)
  self.buffer.write(self.offset_of(index), value)
}

///|
fn check_index(
  shape : Array[Int32],
  index : Array[Int32],
) -> Unit raise VecError {
  if index.length() != shape.length() {
    raise ShapeMismatch(want=shape, got=index)
  }
  for i in 0..<index.length() {
    if index[i] < 0 || index[i] >= shape[i] {
      raise IndexOutOfBounds(index=index[i], len=shape[i])
    }
  }
}

///|
/// The one value of a 0-dimensional array.
pub fn NDArray::item(self : NDArray) -> Number raise VecError {
  if !self.is_scalar() {
    raise ShapeMismatch(want=[], got=self.shape)
  }
  self.buffer.read(self.offset)
}

///|
/// Every element in row-major order, as a fresh array.
pub fn NDArray::to_numbers(self : NDArray) -> Array[Number] raise VecError {
  let out = Array::make(self.size(), Boolean(false))
  let mut at = 0
  for_each_offset(self, fn(offset) raise VecError {
    out[at] = self.buffer.read(offset)
    at = at + 1
  })
  out
}

///|
/// Every element as a double, in row-major order. Complex values lose their imaginary part, which
/// is what NumPy does when a complex array is asked for its `float64` values.
pub fn NDArray::to_doubles(self : NDArray) -> Array[Double] raise VecError {
  let values = self.to_numbers()
  values.map(fn(v) { v.to_double() })
}

///|
/// A contiguous copy, whatever the layout of this view.
pub fn NDArray::copy(self : NDArray) -> NDArray raise VecError {
  let out = alloc(self.shape, self.buffer.dtype)
  if self.is_contiguous() {
    let size = self.buffer.dtype.item_size()
    let from = self.offset * size
    let count = self.size() * size
    for i in 0..<count {
      out.buffer.data[i] = self.buffer.data[from + i]
    }
  } else {
    let mut at = 0
    for_each_offset(self, fn(offset) raise VecError {
      out.buffer.write(at, self.buffer.read(offset))
      at = at + 1
    })
  }
  out
}

///|
/// The product of the axis sizes; an empty shape has one element, the 0-dimensional case.
fn element_count(shape : Array[Int32]) -> Int32 {
  let mut total = 1
  for dim in shape {
    total = total * dim
  }
  total
}

///|
/// The strides of a contiguous array of this shape.
fn contiguous_strides(shape : Array[Int32]) -> Array[Int32] {
  let strides = Array::make(shape.length(), 1)
  if shape.length() == 0 {
    return strides
  }
  let mut run = 1
  let mut i = shape.length() - 1
  while i >= 0 {
    strides[i] = run
    run = run * shape[i]
    i = i - 1
  }
  strides
}

///|
/// Allocates a zeroed contiguous array of `shape`.
fn alloc(shape : Array[Int32], dtype : DType) -> NDArray {
  let count = element_count(shape)
  {
    buffer: Buffer::new(dtype~, len=count),
    shape,
    strides: contiguous_strides(shape),
    offset: 0,
  }
}
