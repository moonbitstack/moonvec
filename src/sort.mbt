// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
/// A sorted copy along one axis, the last by default; NaNs go to the end, as NumPy puts them.
///
/// The sort is stable, so equal values keep the order they had, which `argsort` relies on.
pub fn sort(array : NDArray, axis? : Int32) -> NDArray raise VecError {
  let position = sort_axis(array, axis?)
  let out = array.copy()
  let length = out.shape()[position]
  let step = out.strides()[position]
  for_each_outer(out, position, fn(base, index) raise VecError {
    let values = Array::make(length, Boolean(false))
    for i in 0..<length {
      values[i] = out.buffer.read(base + i * step)
    }
    let sorted = stable_order(values)
    for i in 0..<length {
      out.buffer.write(base + i * step, values[sorted[i]])
    }
    ignore(index)
  })
  out
}

///|
/// The positions that would sort each run along the axis, as `int64`; stable for equal values.
pub fn argsort(array : NDArray, axis? : Int32) -> NDArray raise VecError {
  let position = sort_axis(array, axis?)
  let out = alloc(array.shape(), I64)
  let length = array.shape()[position]
  let step = array.strides()[position]
  let out_step = out.strides()[position]
  for_each_outer(array, position, fn(base, index) raise VecError {
    let values = Array::make(length, Boolean(false))
    for i in 0..<length {
      values[i] = array.buffer.read(base + i * step)
    }
    let order = stable_order(values)
    let target = out.offset() + start_of(out, index, position)
    for i in 0..<length {
      out.buffer.write(target + i * out_step, Signed(order[i].to_int64()))
    }
  })
  out
}

///|
/// The axis a sort runs along: the last one when none is given, and a flattened array when the input
/// has no axes.
fn sort_axis(array : NDArray, axis? : Int32) -> Int32 raise VecError {
  match axis {
    None => {
      if array.ndim() == 0 {
        raise ShapeMismatch(want=array.shape(), got=[])
      }
      array.ndim() - 1
    }
    Some(axis) => axis_position(array, axis)
  }
}

///|
/// The order of `values` from smallest to largest, keeping equal values in their original order.
fn stable_order(values : Array[Number]) -> Array[Int32] {
  let order = Array::makei(values.length(), fn(i) { i })
  order.sort_by(fn(left, right) {
    let by_value = values[left].compare_value(values[right])
    if by_value != 0 {
      by_value
    } else {
      left.compare(right)
    }
  })
  order
}

///|
/// The buffer offset of the position that `for_each_outer` reports as `index`, used to place the
/// results of a sort in the matching slice of the output.
fn start_of(array : NDArray, index : Int32, axis : Int32) -> Int32 {
  let shape = array.shape()
  let strides = array.strides()
  let mut remaining = index
  let mut offset = 0
  let mut i = shape.length() - 1
  while i >= 0 {
    if i != axis {
      let size = shape[i]
      let coordinate = remaining % size
      remaining = remaining / size
      offset = offset + coordinate * strides[i]
    }
    i = i - 1
  }
  offset
}
