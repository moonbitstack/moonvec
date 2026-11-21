// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
/// The shape two arrays take when an elementwise operation meets them.
///
/// The axes are aligned from the right; an axis missing on one side counts as 1, and an axis of 1
/// stretches to the other side's length. Any other pair of different lengths is an error, because
/// there is no honest way to pair those elements up.
pub fn broadcast_shape(
  left : Array[Int32],
  right : Array[Int32],
) -> Array[Int32] raise VecError {
  let ndim = if left.length() > right.length() {
    left.length()
  } else {
    right.length()
  }
  let out = Array::make(ndim, 1)
  let mut i = 0
  while i < ndim {
    let a = axis_from_right(left, i)
    let b = axis_from_right(right, i)
    if a == b || b == 1 {
      out[ndim - 1 - i] = a
    } else if a == 1 {
      out[ndim - 1 - i] = b
    } else {
      raise ShapeMismatch(want=left, got=right)
    }
    i = i + 1
  }
  out
}

///|
/// The length of the `i`-th axis counting from the right, 1 when the shape is shorter.
fn axis_from_right(shape : Array[Int32], i : Int32) -> Int32 {
  let at = shape.length() - 1 - i
  if at < 0 {
    1
  } else {
    shape[at]
  }
}

///|
/// The strides an operand of this shape takes when it is broadcast to `out_shape`: an axis it does
/// not have, or has with length 1, gets stride 0 so every element along that axis reads the same.
pub fn broadcast_strides(
  shape : Array[Int32],
  strides : Array[Int32],
  out_shape : Array[Int32],
) -> Array[Int32] {
  let ndim = out_shape.length()
  let out = Array::make(ndim, 0)
  let mut i = 0
  while i < ndim {
    let at = shape.length() - 1 - i
    if at >= 0 && shape[at] != 1 {
      out[ndim - 1 - i] = strides[at]
    }
    i = i + 1
  }
  out
}

///|
/// Visits every element of `array` in row-major order, passing the buffer offset of each.
///
/// The offsets are produced by an odometer over the shape rather than by recomputing a dot product
/// per element, so a strided or reversed view costs one addition per axis crossed.
pub fn for_each_offset(
  array : NDArray,
  body : (Int32) -> Unit raise VecError,
) -> Unit raise VecError {
  let ndim = array.shape.length()
  if ndim == 0 {
    body(array.offset)
    return
  }
  let mut count = 1
  for dim in array.shape {
    if dim == 0 {
      return
    }
    count = count * dim
  }
  let index = Array::make(ndim, 0)
  let mut at = array.offset
  for _ in 0..<count {
    body(at)
    let mut axis = ndim - 1
    while axis >= 0 {
      index[axis] = index[axis] + 1
      at = at + array.strides[axis]
      if index[axis] < array.shape[axis] {
        break
      }
      at = at - array.strides[axis] * array.shape[axis]
      index[axis] = 0
      axis = axis - 1
    }
  }
}

///|
/// A loop schedule over a shape: the axes that vary between runs, and the length of the innermost
/// run that stays contiguous for every operand.
///
/// Adjacent axes are merged when every operand walks them at a constant step, which turns a
/// contiguous pair of arrays into a single tight loop instead of one call per element. Each run
/// hands the loop body the buffer offsets of all operands at the start of the run.
pub struct Plan {
  outer_sizes : Array[Int32]
  outer_strides : Array[Array[Int32]]
  run : Int32
  run_strides : Array[Int32]
  operands : Int32
  starts : Array[Int32]
}

///|
/// Builds a plan over `out_shape` for operands that have already been broadcast to it.
///
/// Operand 0 is the output; it must be contiguous, which is what `alloc` gives. The rest are read
/// through `strides`, which may hold zeros for broadcast axes.
fn plan_loop(
  out_shape : Array[Int32],
  operands : Array[Array[Int32]],
  starts : Array[Int32],
) -> Plan {
  let ndim = out_shape.length()
  let count = operands.length()
  let sizes : Array[Int32] = []
  let strides : Array[Array[Int32]] = []
  if ndim > 0 {
    let mut size = out_shape[ndim - 1]
    let mut step = Array::make(count, 0)
    for op in 0..<count {
      step[op] = operands[op][ndim - 1]
    }
    let mut axis = ndim - 2
    while axis >= 0 {
      let mut merges = true
      for op in 0..<count {
        if operands[op][axis] != step[op] * size {
          merges = false
        }
      }
      if merges {
        size = size * out_shape[axis]
      } else {
        sizes.push(size)
        strides.push(step)
        size = out_shape[axis]
        let next = Array::make(count, 0)
        for op in 0..<count {
          next[op] = operands[op][axis]
        }
        step = next
      }
      axis = axis - 1
    }
    sizes.push(size)
    strides.push(step)
    sizes.rev_in_place()
    strides.rev_in_place()
  }
  let run = if sizes.length() == 0 { 1 } else { sizes[sizes.length() - 1] }
  let run_strides = if sizes.length() == 0 {
    Array::make(count, 0)
  } else {
    strides[strides.length() - 1]
  }
  let outer_sizes = if sizes.length() <= 1 {
    []
  } else {
    sizes[0:sizes.length() - 1].to_owned()
  }
  let outer_strides = if strides.length() <= 1 {
    []
  } else {
    strides[0:strides.length() - 1].to_owned()
  }
  { outer_sizes, outer_strides, run, run_strides, operands: count, starts, }
}

///|
/// Runs the body once per contiguous run, with the operand offsets at the start of that run.
fn Plan::run(
  self : Plan,
  body : (Array[Int32], Int32) -> Unit raise VecError,
) -> Unit raise VecError {
  // An empty axis means there is no element to visit; without this the loop body would still run
  // once, on a run of length zero, and read the first element of an empty buffer.
  if self.run == 0 {
    return
  }
  for size in self.outer_sizes {
    if size == 0 {
      return
    }
  }
  let bases = self.starts.copy()
  let depth = self.outer_sizes.length()
  if depth == 0 {
    body(bases, self.run)
    return
  }
  let index = Array::make(depth, 0)
  while true {
    body(bases, self.run)
    let mut axis = depth - 1
    let mut advanced = false
    while axis >= 0 {
      index[axis] = index[axis] + 1
      for op in 0..<self.operands {
        bases[op] = bases[op] + self.outer_strides[axis][op]
      }
      if index[axis] < self.outer_sizes[axis] {
        advanced = true
        break
      }
      for op in 0..<self.operands {
        bases[op] = bases[op] -
          self.outer_strides[axis][op] * self.outer_sizes[axis]
      }
      index[axis] = 0
      axis = axis - 1
    }
    if !advanced {
      break
    }
  }
}

///|
/// The plan for an elementwise operation over `inputs` writing into contiguous `out_shape`.
fn plan_elementwise(out_shape : Array[Int32], inputs : Array[NDArray]) -> Plan {
  let operands : Array[Array[Int32]] = [contiguous_strides(out_shape)]
  let starts : Array[Int32] = [0]
  for input in inputs {
    operands.push(broadcast_strides(input.shape, input.strides, out_shape))
    starts.push(input.offset)
  }
  plan_loop(out_shape, operands, starts)
}
