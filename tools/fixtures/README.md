# Parity fixture generator

The generator runs the pinned upstream Python implementation and writes canonical,
versioned JSON to `testdata/parity`. Dependencies and Python are locked by `uv`.

Generate fixtures:

```sh
uv sync --locked --project tools/fixtures
uv run --project tools/fixtures python tools/fixtures/generate.py
```

Verify that a clean regeneration is semantically identical (floating-point
values tolerate platform drift at `1e-12`) and that committed bytes match the
recorded checksums:

```sh
uv run --project tools/fixtures python tools/fixtures/generate.py --check
```

Special IEEE values are encoded as `"NaN"`, `"+Inf"`, and `"-Inf"`. Each fixture
records the upstream package version and source commit. `SHA256SUMS.json` makes
unexpected fixture drift visible in review.
