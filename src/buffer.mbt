// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
/// A flat block of elements of one dtype, which views of an array share.
///
/// The bytes are little endian on every target, so a buffer written on one machine reads back the
/// same on another and a file of vectors can carry raw values. Storage is a byte array rather than
/// a typed one because one code path has to hold every dtype; the accessors below are the only
/// place that knows the widths.
pub struct Buffer {
  data : FixedArray[Byte]
  dtype : DType
  len : Int32
}

///|
/// Allocates `len` elements of `dtype`, zeroed.
pub fn Buffer::new(dtype~ : DType, len~ : Int32) -> Buffer {
  { data: FixedArray::make(len * dtype.item_size(), b'\x00'), dtype, len, }
}

///|
/// Wraps bytes that already hold elements; the length comes from the byte count, so a partial
/// element is an error rather than a silently dropped tail.
pub fn Buffer::from_bytes(
  bytes : Bytes,
  dtype~ : DType,
) -> Buffer raise VecError {
  let size = dtype.item_size()
  let total = bytes.length()
  if total % size != 0 {
    raise ByteCountMismatch(total~, item_size=size)
  }
  { data: bytes.to_fixedarray(), dtype, len: total / size, }
}

///|
/// The element type this buffer stores.
pub fn Buffer::dtype(self : Buffer) -> DType {
  self.dtype
}

///|
/// How many elements the buffer holds.
pub fn Buffer::len(self : Buffer) -> Int32 {
  self.len
}

///|
/// A copy of the bytes, for handing to another library or writing to a file.
pub fn Buffer::to_bytes(self : Buffer) -> Bytes {
  Bytes::from_array(self.data[:])
}

///|
/// The element at `index`, widened to the class its dtype belongs to.
pub fn Buffer::read(self : Buffer, index : Int32) -> Number raise VecError {
  if index < 0 || index >= self.len {
    raise IndexOutOfBounds(index~, len=self.len)
  }
  let at = index * self.dtype.item_size()
  match self.dtype {
    Bool => Boolean(load_uint(self.data, at, 1) != 0UL)
    I8 => Signed(sign_extend(load_uint(self.data, at, 1), 8))
    I16 => Signed(sign_extend(load_uint(self.data, at, 2), 16))
    I32 => Signed(sign_extend(load_uint(self.data, at, 4), 32))
    I64 => Signed(load_uint(self.data, at, 8).reinterpret_as_int64())
    U8 => Unsigned(load_uint(self.data, at, 1))
    U16 => Unsigned(load_uint(self.data, at, 2))
    U32 => Unsigned(load_uint(self.data, at, 4))
    U64 => Unsigned(load_uint(self.data, at, 8))
    F16 =>
      Real(
        @moonmath.Half::from_bits(load_uint(self.data, at, 2).to_uint16())
        .to_float()
        .to_double(),
      )
    F32 => Real(load_f32(self.data, at))
    F64 => Real(load_f64(self.data, at))
    C64 =>
      Complex({ re: load_f32(self.data, at), im: load_f32(self.data, at + 4), })
    C128 =>
      Complex({ re: load_f64(self.data, at), im: load_f64(self.data, at + 8), })
  }
}

///|
/// Stores `value` at `index`, converted the way C converts it to the buffer's dtype.
pub fn Buffer::write(
  self : Buffer,
  index : Int32,
  value : Number,
) -> Unit raise VecError {
  if index < 0 || index >= self.len {
    raise IndexOutOfBounds(index~, len=self.len)
  }
  let at = index * self.dtype.item_size()
  match self.dtype {
    Bool =>
      store_uint(self.data, at, 1, if value.to_bool() { 1UL } else { 0UL })
    I8 => store_uint(self.data, at, 1, value.to_int64().reinterpret_as_uint64())
    I16 =>
      store_uint(self.data, at, 2, value.to_int64().reinterpret_as_uint64())
    I32 =>
      store_uint(self.data, at, 4, value.to_int64().reinterpret_as_uint64())
    I64 =>
      store_uint(self.data, at, 8, value.to_int64().reinterpret_as_uint64())
    U8 => store_uint(self.data, at, 1, value.to_int64().reinterpret_as_uint64())
    U16 =>
      store_uint(self.data, at, 2, value.to_int64().reinterpret_as_uint64())
    U32 =>
      store_uint(self.data, at, 4, value.to_int64().reinterpret_as_uint64())
    U64 =>
      store_uint(self.data, at, 8, value.to_int64().reinterpret_as_uint64())
    F16 =>
      store_uint(
        self.data,
        at,
        2,
        @moonmath.Half::from_float(Float::from_double(value.to_double()))
        .to_bits()
        .to_uint64(),
      )
    F32 => store_f32(self.data, at, value.to_double())
    F64 => store_f64(self.data, at, value.to_double())
    C64 => {
      let c = value.to_complex()
      store_f32(self.data, at, c.re)
      store_f32(self.data, at + 4, c.im)
    }
    C128 => {
      let c = value.to_complex()
      store_f64(self.data, at, c.re)
      store_f64(self.data, at + 8, c.im)
    }
  }
}

///|
/// Sets every element to `value`.
pub fn Buffer::fill(self : Buffer, value : Number) -> Unit raise VecError {
  for i in 0..<self.len {
    self.write(i, value)
  }
}

///|
/// Little endian read of 1 to 8 bytes, zero extended.
fn load_uint(data : FixedArray[Byte], at : Int32, size : Int32) -> UInt64 {
  let mut word = 0UL
  for i in 0..<size {
    word = word | (data[at + i].to_uint64() << (8 * i))
  }
  word
}

///|
/// Little endian write of the low `size` bytes; a negative value keeps its low bits, as C does.
fn store_uint(
  data : FixedArray[Byte],
  at : Int32,
  size : Int32,
  value : UInt64,
) -> Unit {
  for i in 0..<size {
    data[at + i] = ((value >> (8 * i)) & 0xFFUL).to_byte()
  }
}

///|
/// Reads the low `bits` bits as a signed value.
fn sign_extend(value : UInt64, bits : Int32) -> Int64 {
  if bits >= 64 {
    value.reinterpret_as_int64()
  } else {
    let shift = 64 - bits
    (value << shift).reinterpret_as_int64() >> shift
  }
}

///|
fn load_f32(data : FixedArray[Byte], at : Int32) -> Double {
  Float::reinterpret_from_uint(load_uint(data, at, 4).to_uint()).to_double()
}

///|
fn store_f32(data : FixedArray[Byte], at : Int32, value : Double) -> Unit {
  store_uint(
    data,
    at,
    4,
    Float::from_double(value).reinterpret_as_uint().to_uint64(),
  )
}

///|
fn load_f64(data : FixedArray[Byte], at : Int32) -> Double {
  load_uint(data, at, 8).reinterpret_as_double()
}

///|
fn store_f64(data : FixedArray[Byte], at : Int32, value : Double) -> Unit {
  store_uint(data, at, 8, value.reinterpret_as_uint64())
}
