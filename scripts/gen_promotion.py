"""Generate the dtype promotion table from NumPy.

`np.result_type` decides what two arrays of different dtypes turn into when they meet in an
arithmetic expression. The table this prints is checked in, so the library does not need NumPy at
test time; `scripts/gen_vectors.py` writes the same table into the vector file, and a test compares
the library's promotion against it cell by cell.

Usage: python scripts/gen_promotion.py
"""

DTYPES = [
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
]

MOONBIT = {
    "bool": "Bool",
    "int8": "I8",
    "int16": "I16",
    "int32": "I32",
    "int64": "I64",
    "uint8": "U8",
    "uint16": "U16",
    "uint32": "U32",
    "uint64": "U64",
    "float16": "F16",
    "float32": "F32",
    "float64": "F64",
    "complex64": "C64",
    "complex128": "C128",
}


def main():
    import numpy as np

    print(f"// numpy {np.__version__}")
    rows = []
    for a in DTYPES:
        row = []
        for b in DTYPES:
            r = np.result_type(np.dtype(a), np.dtype(b))
            name = r.name
            if name not in MOONBIT:
                raise SystemExit(f"unmapped result type {name} for {a} + {b}")
            row.append(MOONBIT[name])
        rows.append((a, row))
    for a, row in rows:
        print(f"  {a:<11} -> [{', '.join(row)}],")


if __name__ == "__main__":
    main()
