// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
/// What an array operation refuses to do.
///
/// Shapes and axes are checked before any element is touched, so an error never leaves an array
/// half written. Arithmetic that has no answer — an integer divided by zero, an overflowing
/// cast — also lands here rather than returning a number that looks valid.
pub(all) suberror VecError {
  ///
  /// The byte count is not a whole number of elements.
  ByteCountMismatch(total~ : Int32, item_size~ : Int32)
  ///
  /// An index or axis points outside the array.
  IndexOutOfBounds(index~ : Int32, len~ : Int32)
  ///
  /// Two shapes cannot broadcast together, or an operand does not have the shape the operation
  /// needs.
  ShapeMismatch(want~ : Array[Int32], got~ : Array[Int32])
  ///
  /// An axis is outside `0 <= axis < ndim`, or the same axis appears twice.
  AxisOutOfRange(axis~ : Int32, ndim~ : Int32)
  ///
  /// The element count of a reshape does not match the array.
  CannotReshape(size~ : Int32, shape~ : Array[Int32])
  ///
  /// An operation on an empty axis has no result: a maximum of nothing, a mean over zero values.
  EmptyAxis(axis~ : Int32)
  ///
  /// The operation is undefined for the input, such as an integer division by zero.
  Domain(message~ : String)
  ///
  /// A linear algebra routine got a matrix that is not square, or a system with no solution.
  NotSolvable(message~ : String)
}

///|
/// One line naming what was refused, for a caller that holds only the error.
pub fn VecError::message(self : VecError) -> String {
  match self {
    ByteCountMismatch(total~, item_size~) =>
      total.to_string() +
      " bytes is not a whole number of " +
      item_size.to_string() +
      " byte elements"
    IndexOutOfBounds(index~, len~) =>
      "index " + index.to_string() + " is outside 0..<" + len.to_string()
    ShapeMismatch(want~, got~) =>
      "shape [" +
      shape_text(want) +
      "] cannot be used where [" +
      shape_text(got) +
      "] is required"
    AxisOutOfRange(axis~, ndim~) =>
      "axis " + axis.to_string() + " is outside 0..<" + ndim.to_string()
    CannotReshape(size~, shape~) =>
      size.to_string() +
      " elements do not fit shape [" +
      shape_text(shape) +
      "]"
    EmptyAxis(axis~) => "axis " + axis.to_string() + " is empty"
    Domain(message~) => message
    NotSolvable(message~) => message
  }
}

///|
/// `2, 3` for `[2, 3]`, the way a shape reads in a message.
fn shape_text(shape : Array[Int32]) -> String {
  let parts = shape.map(fn(n) { n.to_string() })
  parts.join(", ")
}

///|
pub impl @debug.Debug for VecError with fn to_repr(self) {
  @debug.Repr::literal(self.message())
}
