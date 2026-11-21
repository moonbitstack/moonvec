// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
test "every dtype survives a write and a read" {
  let values : Array[(DType, Number)] = [
    (Bool, Boolean(true)),
    (I8, Signed(-7L)),
    (I16, Signed(-300L)),
    (I32, Signed(70000L)),
    (I64, Signed(-9007199254740993L)),
    (U8, Unsigned(200UL)),
    (U16, Unsigned(60000UL)),
    (U32, Unsigned(4000000000UL)),
    (U64, Unsigned(18446744073709551615UL)),
    (F16, Real(1.5)),
    (F32, Real(0.25)),
    (F64, Real(-2.5)),
    (C64, complex_number(1.5, -2.5)),
    (C128, complex_number(-0.125, 8.0)),
  ]
  for entry in values {
    let (dtype, value) = entry
    let buffer = Buffer::new(dtype~, len=2)
    buffer.write(1, value)
    assert_eq(buffer.read(1), value.cast(dtype))
    assert_eq(buffer.read(0).to_double(), 0.0)
  }
}

///|
test "a half precision value keeps the bits it was given" {
  let buffer = Buffer::new(dtype=F16, len=1)
  buffer.write(0, Real(0.1))
  // 0.1 is not representable; the stored value is the nearest half, which reads back as itself.
  let stored = buffer.read(0)
  let again = Buffer::new(dtype=F16, len=1)
  again.write(0, stored)
  assert_eq(again.read(0), stored)
  assert_eq(buffer.to_bytes().length(), 2)
}

///|
test "bytes hold the elements little endian" {
  let buffer = Buffer::new(dtype=I32, len=1)
  buffer.write(0, Signed(0x01020304L))
  let bytes = buffer.to_bytes()
  assert_eq(bytes.at(0), b'\x04')
  assert_eq(bytes.at(1), b'\x03')
  assert_eq(bytes.at(2), b'\x02')
  assert_eq(bytes.at(3), b'\x01')
}

///|
test "a buffer round-trips through its bytes" {
  let buffer = Buffer::new(dtype=F64, len=3)
  buffer.write(0, Real(0.5))
  buffer.write(1, Real(-1.25))
  buffer.write(2, Real(3.0))
  let again = Buffer::from_bytes(buffer.to_bytes(), dtype=F64)
  assert_eq(again.len(), 3)
  assert_eq(again.read(0), Real(0.5))
  assert_eq(again.read(1), Real(-1.25))
  assert_eq(again.read(2), Real(3.0))
}

///|
test "a byte count that is not a whole number of elements is refused" {
  let bytes = Bytes::from_array([b'\x00', b'\x01', b'\x02'])
  let refused : Result[Unit, VecError] = try {
    let _ = Buffer::from_bytes(bytes, dtype=I16)
    Ok(())
  } catch {
    error => Err(error)
  }
  assert_eq(refused is Err(ByteCountMismatch(_)), true)
}

///|
test "reading past the end of a buffer is refused" {
  let buffer = Buffer::new(dtype=F64, len=2)
  let refused : Result[Unit, VecError] = try {
    ignore(buffer.read(2))
    Ok(())
  } catch {
    error => Err(error)
  }
  assert_eq(refused is Err(IndexOutOfBounds(_)), true)
}

///|
test "filling a buffer sets every element" {
  let buffer = Buffer::new(dtype=I8, len=4)
  buffer.fill(Signed(-2L))
  assert_eq(buffer.read(3), Signed(-2L))
}
