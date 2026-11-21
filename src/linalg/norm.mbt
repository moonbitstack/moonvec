// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
/// How a norm measures a vector or a matrix.
pub(all) enum Norm {
  ///
  /// For a vector the Euclidean length; for a matrix the root of the sum of squared entries.
  Frobenius
  ///
  /// For a vector the sum of absolute values; for a matrix the largest column sum.
  One
  ///
  /// For a vector the Euclidean length; for a matrix the largest singular value.
  Two
  ///
  /// For a vector the largest absolute value; for a matrix the largest row sum.
  Infinity
  ///
  /// The smallest absolute value of a vector.
  NegInfinity
  ///
  /// The `p`-th root of the sum of the absolute values raised to `p`.
  P(Double)
}

///|
/// The norm of a vector, or of every vector along one axis.
///
/// Without `axis` the whole array is read as one vector and the answer is 0-dimensional; with it, each
/// line along that axis is measured and the axis disappears.
pub fn norm(
  a : @v.NDArray,
  order~ : Norm = Frobenius,
  axis? : Int32,
) -> @v.NDArray raise @v.VecError {
  match axis {
    None => {
      // A matrix read as a whole has its own norms: the largest column sum, the largest row sum, the
      // largest singular value. A single axis or none on a vector uses the vector norms.
      if a.ndim() == 2 {
        return scalar(matrix_norm(a, order~), real_dtype(a))
      }
      let values = a.to_doubles()
      scalar(norm_of(values, order), real_dtype(a))
    }
    Some(axis) => {
      let position = axis_position(a, axis)
      let values = a.to_doubles()
      let shape = a.shape()
      let length = shape[position]
      let mut outer = 1
      for i in 0..<shape.length() {
        if i != position {
          outer = outer * shape[i]
        }
      }
      // The data reads row-major, so a line along `position` strides by that axis's stride in the
      // flat order.
      let mut stride = 1
      let mut axis = position + 1
      while axis < shape.length() {
        stride = stride * shape[axis]
        axis = axis + 1
      }

      let out : Array[Double] = []
      for line in 0..<outer {
        let block = line / stride
        let within = line % stride
        let line_values : Array[Double] = []
        for i in 0..<length {
          line_values.push(
            values[block * length * stride + i * stride + within],
          )
        }
        out.push(norm_of(line_values, order))
      }
      let reduced : Array[Int32] = []
      for i in 0..<shape.length() {
        if i != position {
          reduced.push(shape[i])
        }
      }
      @v.from_doubles(out, reduced, dtype=real_dtype(a))
    }
  }
}

///|
/// The norm of a matrix read as a whole.
pub fn matrix_norm(
  a : @v.NDArray,
  order~ : Norm = Frobenius,
) -> Double raise @v.VecError {
  let shape = a.shape()
  if shape.length() != 2 {
    raise @v.ShapeMismatch(want=[2], got=shape)
  }
  let rows = shape[0]
  let columns = shape[1]
  let values = a.to_doubles()
  match order {
    Frobenius => norm_of(values, Frobenius)
    One => {
      let mut largest = 0.0
      for column in 0..<columns {
        let mut total = 0.0
        for row in 0..<rows {
          total = total + values[row * columns + column].abs()
        }
        largest = largest.max(total)
      }
      largest
    }
    Infinity => {
      let mut largest = 0.0
      for row in 0..<rows {
        let mut total = 0.0
        for column in 0..<columns {
          total = total + values[row * columns + column].abs()
        }
        largest = largest.max(total)
      }
      largest
    }
    Two => {
      let (_, singular, _) = svd(a)
      let values = singular.to_numbers()
      let mut largest = 0.0
      for value in values {
        largest = largest.max(value.to_double())
      }
      largest
    }
    NegInfinity | P(_) => norm_of(values, order)
  }
}

///|
/// The norm of a plain list of values.
fn norm_of(values : Array[Double], order : Norm) -> Double {
  match order {
    Frobenius | Two => {
      let mut total = 0.0
      for value in values {
        total = total + value * value
      }
      total.sqrt()
    }
    One => {
      let mut total = 0.0
      for value in values {
        total = total + value.abs()
      }
      total
    }
    Infinity => {
      let mut largest = 0.0
      for value in values {
        largest = largest.max(value.abs())
      }
      largest
    }
    NegInfinity => {
      let mut smallest = 1.0 / 0.0
      for value in values {
        smallest = smallest.min(value.abs())
      }
      smallest
    }
    P(exponent) =>
      if exponent == 0.0 {
        let mut count = 0.0
        for value in values {
          if value != 0.0 {
            count = count + 1.0
          }
        }
        count
      } else {
        let mut total = 0.0
        for value in values {
          total = total + @math.pow(value.abs(), exponent)
        }
        @math.pow(total, 1.0 / exponent)
      }
  }
}

///|
fn axis_position(a : @v.NDArray, axis : Int32) -> Int32 raise @v.VecError {
  let ndim = a.ndim()
  let position = if axis < 0 { axis + ndim } else { axis }
  if position < 0 || position >= ndim {
    raise @v.AxisOutOfRange(axis~, ndim~)
  }
  position
}
