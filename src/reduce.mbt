// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
/// The dtype a reduction produces: a sum of booleans counts in `int64`, and everything else keeps
/// the array's dtype.
fn reduction_dtype(dtype : DType, op : ReduceOp) -> DType {
  match op {
    Sum | Product =>
      if dtype.is_bool() || dtype == I8 || dtype == I16 || dtype == I32 {
        I64
      } else if dtype == U8 || dtype == U16 || dtype == U32 {
        U64
      } else {
        dtype
      }
    Minimum | Maximum => dtype
    Any | All => Bool
  }
}

///|
/// Sums the elements, either all of them or along one axis.
///
/// A full reduction returns a 0-dimensional array; `keepdims` keeps the reduced axis with length 1
/// so the result still broadcasts against the input.
pub fn sum(
  a : NDArray,
  axis? : Int32,
  keepdims~ : Bool = false,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  reduce(a, Sum, axis?, keepdims, pick_kernel(kernel))
}

///|
/// Multiplies the elements.
pub fn prod(
  a : NDArray,
  axis? : Int32,
  keepdims~ : Bool = false,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  reduce(a, Product, axis?, keepdims, pick_kernel(kernel))
}

///|
/// The smallest element; an empty axis has no smallest element and is refused.
pub fn min(
  a : NDArray,
  axis? : Int32,
  keepdims~ : Bool = false,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  reduce(a, Minimum, axis?, keepdims, pick_kernel(kernel))
}

///|
/// The largest element.
pub fn max(
  a : NDArray,
  axis? : Int32,
  keepdims~ : Bool = false,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  reduce(a, Maximum, axis?, keepdims, pick_kernel(kernel))
}

///|
/// True where any element is true.
pub fn any(
  a : NDArray,
  axis? : Int32,
  keepdims~ : Bool = false,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  reduce(a, Any, axis?, keepdims, pick_kernel(kernel))
}

///|
/// True where every element is true; a full reduction over an empty array is true.
pub fn all(
  a : NDArray,
  axis? : Int32,
  keepdims~ : Bool = false,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  reduce(a, All, axis?, keepdims, pick_kernel(kernel))
}

///|
/// The arithmetic mean: an integer array answers in `float64`, and an empty axis gives NaN rather
/// than an error, since there is a count to divide by, namely zero.
pub fn mean(
  a : NDArray,
  axis? : Int32,
  keepdims~ : Bool = false,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  let totals = sum(a, axis?, keepdims~, kernel?)
  let count = count_along(a, axis)
  let dtype = if totals.dtype().is_exact() { F64 } else { totals.dtype() }
  let out = alloc(totals.shape(), dtype)
  let mut at = 0
  for_each_offset(totals, fn(offset) raise VecError {
    out.buffer.write(
      at,
      Real(totals.buffer.read(offset).to_double() / count.to_double()),
    )
    at = at + 1
  })
  out
}

///|
/// The variance with `ddof` degrees of freedom removed from the count: the population variance by
/// default, the sample variance with `ddof=1`.
pub fn variance(
  a : NDArray,
  axis? : Int32,
  ddof~ : Int32 = 0,
  keepdims~ : Bool = false,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  let count = count_along(a, axis)
  if count - ddof <= 0 {
    raise Domain(
      message="variance needs more values than the degrees of freedom removed",
    )
  }
  let average = mean(a, axis?, keepdims=true, kernel?)
  let spread = subtract(a, average, kernel?)
  let squared = multiply(spread, spread, kernel?)
  let totals = sum(squared, axis?, keepdims=true, kernel?)
  let dtype = if totals.dtype().is_exact() { F64 } else { totals.dtype() }
  let values = totals.to_numbers()
  let out = alloc(reduced_shape(a, axis, keepdims), dtype)
  for i in 0..<values.length() {
    out.buffer.write(
      i,
      Real(values[i].to_double() / (count - ddof).to_double()),
    )
  }
  out
}

///|
/// The standard deviation, the square root of the variance.
pub fn std(
  a : NDArray,
  axis? : Int32,
  ddof~ : Int32 = 0,
  keepdims~ : Bool = false,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  let spread = variance(a, axis?, ddof~, keepdims~, kernel?)
  let values = spread.to_numbers()
  let out = alloc(spread.shape(), spread.dtype())
  for i in 0..<values.length() {
    out.buffer.write(i, Real(values[i].to_double().sqrt()))
  }
  out
}

///|
/// The index of the smallest element along the axis; the first one wins a tie.
pub fn argmin(
  a : NDArray,
  axis? : Int32,
  keepdims~ : Bool = false,
) -> NDArray raise VecError {
  arg_reduce(a, axis?, keepdims, false)
}

///|
/// The index of the largest element along the axis; the first one wins a tie.
pub fn argmax(
  a : NDArray,
  axis? : Int32,
  keepdims~ : Bool = false,
) -> NDArray raise VecError {
  arg_reduce(a, axis?, keepdims, true)
}

///|
/// The shape a reduction leaves behind: the axis removed, or kept with length 1, or every axis kept
/// when the reduction covered the whole array.
fn reduced_shape(
  a : NDArray,
  axis : Int32?,
  keepdims : Bool,
) -> Array[Int32] raise VecError {
  match axis {
    None => if keepdims { Array::make(a.ndim(), 1) } else { [] }
    Some(axis) => {
      let position = check_axis(a, axis)
      if keepdims {
        let shape = a.shape()
        shape[position] = 1
        shape
      } else {
        remove_axis(a.shape(), position)
      }
    }
  }
}

///|
/// How many elements a reduction along `axis` folds; the whole array when `axis` is absent.
fn count_along(a : NDArray, axis : Int32?) -> Int32 raise VecError {
  match axis {
    None => a.size()
    Some(axis) => a.shape()[check_axis(a, axis)]
  }
}

///|
/// The body of the reductions: either one value over the whole array, or one per position along the
/// removed axis.
fn reduce(
  a : NDArray,
  op : ReduceOp,
  axis? : Int32,
  keepdims : Bool,
  kernel : &Kernel,
) -> NDArray raise VecError {
  let dtype = reduction_dtype(a.dtype(), op)
  let identity = reduce_identity(op, a.dtype())
  match axis {
    None => {
      let mut total = identity
      let plan = plan_loop(a.shape(), [contiguous_strides(a.shape())], [
        a.offset(),
      ])
      plan.run(fn(bases, len) raise VecError {
        total = kernel.reduce(
          op,
          dtype,
          a.buffer,
          bases[0],
          plan.run_strides[0],
          len,
          total,
        )
      })
      let shape = if keepdims { Array::make(a.ndim(), 1) } else { [] }
      let out = alloc(shape, dtype)
      out.buffer.write(0, total)
      out
    }
    Some(axis) => {
      let position = check_axis(a, axis)
      let length = a.shape()[position]
      if length == 0 {
        match op {
          Minimum | Maximum => raise EmptyAxis(axis=position)
          _ => ()
        }
      }
      let out_shape = if keepdims {
        let shape = a.shape()
        shape[position] = 1
        shape
      } else {
        remove_axis(a.shape(), position)
      }
      let out = alloc(out_shape, dtype)
      let mut at = 0
      for_each_outer(a, position, fn(base, index) raise VecError {
        let value = kernel.reduce(
          op,
          dtype,
          a.buffer,
          base,
          a.strides()[position],
          length,
          identity,
        )
        out.buffer.write(at, value)
        at = at + 1
        ignore(index)
      })
      out
    }
  }
}

///|
fn arg_reduce(
  a : NDArray,
  axis? : Int32,
  keepdims : Bool,
  largest : Bool,
) -> NDArray raise VecError {
  match axis {
    None => {
      if a.size() == 0 {
        raise EmptyAxis(axis=0)
      }
      let mut best = 0
      let mut best_value = Boolean(false)
      let mut first = true
      let mut at = 0
      for_each_offset(a, fn(offset) raise VecError {
        let value = a.buffer.read(offset)
        let better = if first {
          true
        } else if largest {
          value.compare_value(best_value) > 0
        } else {
          value.compare_value(best_value) < 0
        }
        if better {
          best = at
          best_value = value
          first = false
        }
        at = at + 1
      })
      let shape = if keepdims { Array::make(a.ndim(), 1) } else { [] }
      let out = alloc(shape, I64)
      out.buffer.write(0, Signed(best.to_int64()))
      out
    }
    Some(axis) => {
      let position = check_axis(a, axis)
      let length = a.shape()[position]
      if length == 0 {
        raise EmptyAxis(axis=position)
      }
      let out_shape = if keepdims {
        let shape = a.shape()
        shape[position] = 1
        shape
      } else {
        remove_axis(a.shape(), position)
      }
      let out = alloc(out_shape, I64)
      let stride = a.strides()[position]
      let mut at = 0
      for_each_outer(a, position, fn(base, index) raise VecError {
        let mut best = 0
        let mut best_value = a.buffer.read(base)
        for i in 1..<length {
          let value = a.buffer.read(base + i * stride)
          let better = if largest {
            value.compare_value(best_value) > 0
          } else {
            value.compare_value(best_value) < 0
          }
          if better {
            best = i
            best_value = value
          }
        }
        out.buffer.write(at, Signed(best.to_int64()))
        at = at + 1
        ignore(index)
      })
      out
    }
  }
}

///|
/// Checks that `axis` names one of the axes, negative counting from the end, and returns its
/// position from the front.
fn check_axis(a : NDArray, axis : Int32) -> Int32 raise VecError {
  let ndim = a.ndim()
  let position = if axis < 0 { axis + ndim } else { axis }
  if position < 0 || position >= ndim {
    raise AxisOutOfRange(axis~, ndim~)
  }
  position
}

///|
/// The shape without one axis.
fn remove_axis(shape : Array[Int32], position : Int32) -> Array[Int32] {
  let out : Array[Int32] = []
  for i in 0..<shape.length() {
    if i != position {
      out.push(shape[i])
    }
  }
  out
}

///|
/// Visits the start of every slice along `axis`, in row-major order over the other axes, passing the
/// buffer offset of the slice's first element.
fn for_each_outer(
  a : NDArray,
  axis : Int32,
  body : (Int32, Int32) -> Unit raise VecError,
) -> Unit raise VecError {
  let ndim = a.ndim()
  let dims : Array[Int32] = []
  let strides : Array[Int32] = []
  for i in 0..<ndim {
    if i != axis {
      dims.push(a.shape()[i])
      strides.push(a.strides()[i])
    }
  }
  for dim in dims {
    if dim == 0 {
      return
    }
  }
  let depth = dims.length()
  let index = Array::make(depth, 0)
  let mut base = a.offset()
  let mut at = 0
  while true {
    body(base, at)
    at = at + 1
    let mut moved = false
    let mut i = depth - 1
    while i >= 0 {
      index[i] = index[i] + 1
      base = base + strides[i]
      if index[i] < dims[i] {
        moved = true
        break
      }
      base = base - strides[i] * dims[i]
      index[i] = 0
      i = i - 1
    }
    if !moved {
      break
    }
  }
}
