# GASmith Cross-Check Fixtures

This directory holds JSON files that encode reference Clifford products exported
from GASmith.  The Tensorsmith test suite loads them to validate the Clifford
normalization against an independent C++ implementation.

Tests skip gracefully if the file is absent, so CI passes without them.

---

## Format

`gasmith_Cl300.json` — Cl(3,0,0) product table

```json
{
  "algebra":   "Cl(3,0,0)",
  "n":         3,
  "signature": { "p": 3, "q": 0, "r": 0 },
  "products": [
    {
      "a":      [1],
      "b":      [1],
      "result": [[ [], 1 ]]
    },
    {
      "a":      [1],
      "b":      [2],
      "result": [[ [1, 2], 1 ]]
    },
    {
      "a":      [2],
      "b":      [1],
      "result": [[ [1, 2], -1 ]]
    }
  ]
}
```

Fields:
- `a`, `b` — strictly-increasing index arrays (1-based) for the input basis blades.
  Grade-0 scalars use `[]`.
- `result` — array of `[index, coefficient]` pairs.  Use integer coefficients for
  exact comparison; the test converts them to `Rational{BigInt}`.

---

## How to export from GASmith

Add a small export function to GASmith's test suite or a standalone script:

```cpp
#include <nlohmann/json.hpp>   // or any JSON library
#include "GASmith/GASmith.h"

// For each blade pair (A, B) in the algebra, compute A*B and record the terms.
// Blade index to sorted Int vector: iterate over set bits of the bitmask.
//   bitmask 0b001 = 1 → [1]
//   bitmask 0b011 = 3 → [1, 2]
//   bitmask 0b110 = 6 → [2, 3]
```

For Cl(3,0,0) it is sufficient to export all 64 products of the 8 grade-0…3 basis
blades.  Run the script, save the output to this directory as `gasmith_Cl300.json`.
