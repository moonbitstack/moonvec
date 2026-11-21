// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
/// One case from the generated vector file: the operands, the operation, and the answer NumPy gave.
struct VectorCase {
  name : String
  op : String
  tolerance : Double
  inputs : Array[NDArray]
  outputs : Array[NDArray]
}

///|
test "every generated case answers what NumPy answered" {
  let lines = vector_text().split("\n").collect()
  let cases : Array[VectorCase] = []
  let mut pending_inputs : Array[NDArray] = []
  let mut pending_outputs : Array[NDArray] = []
  let mut pending_name = ""
  let mut pending_op = ""
  let mut pending_tolerance = 0.0
  let mut seen_promotions = false
  let failures : Array[String] = []
  for line in lines {
    let text = line.to_owned()
    if text == "" || text.has_prefix("#") {
      continue
    }
    let fields = text.split(" ").collect().map(fn(field) { field.to_owned() })
    match fields[0] {
      "promotions" => ()
      "promote" => {
        let left = dtype_or_zero(fields[1])
        let right = dtype_or_zero(fields[2])
        let expected = dtype_or_zero(fields[3])
        seen_promotions = true
        if promote(left, right) != expected {
          failures.push(
            "promote " +
            left.name() +
            " " +
            right.name() +
            ": got " +
            promote(left, right).name() +
            ", want " +
            expected.name(),
          )
        }
      }
      "cases" => ()
      "case" => {
        pending_name = text[5:].to_owned()
        pending_inputs = []
        pending_outputs = []
        pending_op = ""
        pending_tolerance = 0.0
      }
      "op" => pending_op = text[3:].to_owned()
      "tol" => pending_tolerance = parse_double(fields[1])
      "in" => pending_inputs.push(decode_array(fields[1], fields[2], fields[3]))
      "out" =>
        pending_outputs.push(decode_array(fields[1], fields[2], fields[3]))
      "end" =>
        cases.push({
          name: pending_name,
          op: pending_op,
          tolerance: pending_tolerance,
          inputs: pending_inputs,
          outputs: pending_outputs,
        })
      _ => failures.push("unknown vector line: " + text)
    }
  }
  assert_eq(seen_promotions, true)
  for case in cases {
    let actual : Array[NDArray] = run_vector_case(case) catch {
      error => {
        failures.push(case.name + ": raised " + error.message())
        []
      }
    }
    if actual.length() != case.outputs.length() {
      failures.push(
        case.name +
        ": " +
        actual.length().to_string() +
        " results, want " +
        case.outputs.length().to_string(),
      )
      continue
    }
    for i in 0..<actual.length() {
      match mismatch(actual[i], case.outputs[i], case.tolerance) {
        Some(reason) => failures.push(case.name + ": " + reason)
        None => ()
      }
    }
  }
  if failures.length() > 0 {
    let shown = if failures.length() > 20 { 20 } else { failures.length() }
    let mut report = failures.length().to_string() +
      " of " +
      cases.length().to_string() +
      " cases differ from NumPy:\n"
    for i in 0..<shown {
      report = report + "  " + failures[i] + "\n"
    }
    fail(report)
  }
}

///|
/// Reports how two arrays differ, or nothing when they agree.
fn mismatch(
  actual : NDArray,
  expected : NDArray,
  tolerance : Double,
) -> String? {
  if actual.shape() != expected.shape() {
    return Some(
      "shape " +
      shape_text(actual.shape()) +
      ", want " +
      shape_text(expected.shape()),
    )
  }
  if actual.dtype() != expected.dtype() {
    return Some(
      "dtype " + actual.dtype().name() + ", want " + expected.dtype().name(),
    )
  }
  let left = match (Ok(actual.to_numbers()) catch { error => Err(error) }) {
    Ok(values) => values
    Err(error) => return Some("reading the result raised " + error.message())
  }
  let right = expected.to_numbers() catch {
    error => return Some("reading the vector raised " + error.message())
  }
  for i in 0..<left.length() {
    if !within(left[i], right[i], tolerance) {
      return Some(
        "element " +
        i.to_string() +
        ": " +
        left[i].to_string() +
        ", want " +
        right[i].to_string() +
        " (tol " +
        format_real(tolerance) +
        ")",
      )
    }
  }
  None
}

///|
/// Whether two values agree: exactly when `tolerance` is zero, which still counts a NaN as equal to
/// a NaN and a zero of either sign as the same value.
fn within(actual : Number, expected : Number, tolerance : Double) -> Bool {
  match (actual, expected) {
    (Complex(a), Complex(b)) => {
      if a.re.is_nan() || a.im.is_nan() || b.re.is_nan() || b.im.is_nan() {
        return a.re.is_nan() && a.im.is_nan() && b.re.is_nan() && b.im.is_nan()
      }
      if a.re == b.re && a.im == b.im {
        return true
      }
      if tolerance == 0.0 {
        return false
      }
      // The distance between two complex values is measured against their modulus, not against each
      // part, or a small imaginary part would be allowed less error than the number has.
      let distance = @math.hypot(a.re - b.re, a.im - b.im)
      let scale = @math.hypot(a.re, a.im).max(@math.hypot(b.re, b.im)).max(1.0)
      distance <= tolerance * scale
    }
    _ => close(actual.to_double(), expected.to_double(), tolerance)
  }
}

///|
fn close(actual : Double, expected : Double, tolerance : Double) -> Bool {
  if actual.is_nan() || expected.is_nan() {
    return actual.is_nan() && expected.is_nan()
  }
  if actual == expected {
    return true
  }
  if tolerance == 0.0 {
    return false
  }
  let scale = actual.abs().max(expected.abs()).max(1.0)
  (actual - expected).abs() <= tolerance * scale
}

///|
/// The dtype named on a line, falling back to `float64` so that a malformed file reports a failure
/// instead of stopping the test.
fn dtype_or_zero(name : String) -> DType {
  match DType::parse(name) {
    Some(dtype) => dtype
    None => F64
  }
}

///|
/// Rebuilds an array from a vector line: the dtype name, the axis sizes, and the bytes as hex.
fn decode_array(
  dtype_name : String,
  dims_text : String,
  payload : String,
) -> NDArray {
  let dtype = dtype_or_zero(dtype_name)
  let shape : Array[Int32] = if dims_text == "-" {
    []
  } else {
    dims_text.split("x").collect().map(fn(part) { parse_int(part) })
  }
  let bytes = decode_hex(payload)
  let buffer = Buffer::from_bytes(bytes, dtype~) catch {
    error => abort(error.message())
  }
  NDArray::new(buffer~, shape~) catch {
    error => abort(error.message())
  }
}

///|
fn decode_hex(text : String) -> Bytes {
  if text == "-" {
    return Bytes::from_array([])
  }
  let out : Array[Byte] = []
  let mut i = 0
  while i < text.length() {
    let high = hex_digit(text[i])
    let low = hex_digit(text[i + 1])
    out.push(((high << 4) | low).to_byte())
    i = i + 2
  }
  Bytes::from_array(out[:])
}

///|
fn hex_digit(code : UInt16) -> Int32 {
  let value = code.to_int()
  if value >= 48 && value <= 57 {
    value - 48
  } else if value >= 97 && value <= 102 {
    value - 87
  } else {
    0
  }
}

///|
/// A whole number from the vector file; the generator writes plain decimal.
fn parse_int(text : StringView) -> Int32 {
  let mut value = 0
  let mut negative = false
  for i in 0..<text.length() {
    let character = text[i]
    if i == 0 && character == '-' {
      negative = true
    } else {
      value = value * 10 + (character.to_int() - '0'.to_int())
    }
  }
  if negative {
    -value
  } else {
    value
  }
}

///|
/// A decimal double from the vector file, written by Python's `repr`.
fn parse_double(text : StringView) -> Double {
  let parsed = @json.parse(text) catch { _ => return 0.0 }
  match parsed {
    Number(value, ..) => value
    _ => 0.0
  }
}
