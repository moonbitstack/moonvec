// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
test "every linear algebra case answers what NumPy answered" {
  let lines = vector_text().split("\n").collect()
  let cases : Array[VectorCase] = []
  let inputs : Array[@v.NDArray] = []
  let outputs : Array[@v.NDArray] = []
  let mut name = ""
  let mut operation = ""
  let mut tolerance = 0.0
  let failures : Array[String] = []
  for line in lines {
    let text = line.to_owned()
    if text == "" || text.has_prefix("#") {
      continue
    }
    let fields = text.split(" ").collect().map(fn(field) { field.to_owned() })
    match fields[0] {
      "cases" => ()
      "case" => {
        name = text[5:].to_owned()
        operation = ""
        tolerance = 0.0
        inputs.clear()
        outputs.clear()
      }
      "op" => operation = text[3:].to_owned()
      "tol" => tolerance = parse_double(fields[1])
      "in" => inputs.push(decode_array(fields[1], fields[2], fields[3]))
      "out" => outputs.push(decode_array(fields[1], fields[2], fields[3]))
      "end" =>
        cases.push({
          name,
          op: operation,
          tolerance,
          inputs: inputs.copy(),
          outputs: outputs.copy(),
        })
      "promotions" | "promote" => ()
      _ => failures.push("unknown vector line: " + text)
    }
  }
  for case in cases {
    let actual : Array[@v.NDArray] = run_linalg_case(case) catch {
      error => {
        failures.push(case.name + ": raised " + error.message())
        []
      }
    }
    if actual.length() != case.outputs.length() {
      failures.push(case.name + ": " + actual.length().to_string() + " results")
      continue
    }
    for i in 0..<actual.length() {
      let expected = case.outputs[i]
      if actual[i].shape() != expected.shape() {
        failures.push(case.name + ": shape " + shape_text(actual[i].shape()))
      } else if actual[i].dtype() != expected.dtype() {
        failures.push(
          case.name +
          ": dtype " +
          actual[i].dtype().name() +
          ", want " +
          expected.dtype().name(),
        )
      } else {
        let left = actual[i].to_doubles()
        let right = expected.to_doubles()
        for k in 0..<left.length() {
          let scale = left[k].abs().max(right[k].abs()).max(1.0)
          if (left[k] - right[k]).abs() > case.tolerance * scale {
            failures.push(
              case.name +
              ": element " +
              k.to_string() +
              " is " +
              @v.format_real(left[k]) +
              ", want " +
              @v.format_real(right[k]),
            )
            break
          }
        }
      }
    }
  }
  if failures.length() > 0 {
    let mut report = failures.length().to_string() +
      " of " +
      cases.length().to_string() +
      " linear algebra cases differ from NumPy:\n"
    for failure in failures {
      report = report + "  " + failure + "\n"
    }
    fail(report)
  }
}

///|
/// The shape of a case: the operation with its parameters after colons.
struct VectorCase {
  name : String
  op : String
  tolerance : Double
  inputs : Array[@v.NDArray]
  outputs : Array[@v.NDArray]
}

///|
fn run_linalg_case(case : VectorCase) -> Array[@v.NDArray] raise @v.VecError {
  let pieces = case.op.split(":").collect().map(fn(piece) { piece.to_owned() })
  let operation = pieces[0]
  let parameters = pieces[1:].to_owned()
  let results : Array[@v.NDArray] = []
  let left = case.inputs[0]
  let right = if case.inputs.length() > 1 { case.inputs[1] } else { left }
  match operation {
    "solve" => results.push(solve(left, right))
    "inv" => results.push(inv(left))
    "det" => results.push(@v.from_doubles([det(left)], []))
    "cholesky" => results.push(cholesky(left))
    "lstsq" => results.push(lstsq(left, right))
    "norm" => {
      let order = norm_order(parameters[0])
      let axis = axis_setting(parameters)
      results.push(norm(left, order~, axis?))
    }
    "matrix_power" =>
      results.push(matrix_power(left, exponent=parse_int(parameters[0])))
    "kron" => results.push(kron(left, right))
    _ => results.push(@v.zeros([0]))
  }
  results
}

///|
/// The norm order the generator wrote: `fro`, `1`, `2`, `inf`, `-inf` or a number.
fn norm_order(text : String) -> Norm {
  match text {
    "fro" => Frobenius
    "1" => One
    "2" => Two
    "inf" => Infinity
    "-inf" => NegInfinity
    _ => P(parse_double(text))
  }
}

///|
fn axis_setting(parameters : Array[String]) -> Int32? {
  for parameter in parameters {
    if parameter.has_prefix("axis=") {
      return Some(parse_int(parameter[5:].to_owned()))
    }
  }
  None
}

///|
fn decode_array(
  dtype_name : String,
  dims_text : String,
  payload : String,
) -> @v.NDArray {
  let dtype = match @v.DType::parse(dtype_name) {
    Some(dtype) => dtype
    None => @v.F64
  }
  let shape : Array[Int32] = if dims_text == "-" {
    []
  } else {
    dims_text.split("x").collect().map(fn(part) { parse_int(part) })
  }
  let bytes = decode_hex(payload)
  let buffer = @v.Buffer::from_bytes(bytes, dtype~) catch {
    error => abort(error.message())
  }
  @v.NDArray::new(buffer~, shape~) catch {
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
fn parse_int(text : StringView) -> Int32 {
  let mut value = 0
  let mut negative = false
  for i in 0..<text.length() {
    let code = text[i].to_int()
    if i == 0 && code == 45 {
      negative = true
    } else {
      value = value * 10 + (code - 48)
    }
  }
  if negative {
    -value
  } else {
    value
  }
}

///|
fn parse_double(text : StringView) -> Double {
  let parsed = @json.parse(text) catch { _ => return 0.0 }
  match parsed {
    Number(value, ..) => value
    _ => 0.0
  }
}

///|
fn shape_text(shape : Array[Int32]) -> String {
  "[" + shape.map(fn(dim) { dim.to_string() }).join(", ") + "]"
}
