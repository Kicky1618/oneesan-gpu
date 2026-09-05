# Row-6 exact rational certificate

`row6_rational_dump.txt.xz` is the compressed exact-rational certificate used to generate the
production 20-prime row-6 coefficient table. The uncompressed dump SHA-256 is
`a7011c70cb28d556d3155c525a12e6f4e8b23d3a6e5a154cd3f4974a8311cf80`; the compressed XZ file
SHA-256 is `58c12e5cab188c6118637266b4627a8101c0c8f1ae5f8227a385cea0cf96fb1f`.

Verify all exact rational coefficients against both the independent mod-1,000,000,007 automaton and
the 20 production CRT rows with:

```bash
python3 scripts/tools/verify_row6_crt20.py
```

Regenerate the production coefficient header with:

```bash
python3 scripts/tools/gen_row6_crt20.py
```

The certificate is a reproducibility artifact for the generated row-6 automaton. It is not a Lean
proof of the automaton-learning/elimination procedure itself.
The n=27 production build records this certificate, `scripts/tools/gen_row6_crt20.py`,
`scripts/tools/verify_row6_crt20.py`, and `scripts/solve/path_bound.py` in the V2 build-provenance
`auxiliary_dependencies` list. Consequently an archived exact-result sidecar identifies not only the
generated header bytes but also the exact certificate and code used to derive them. When
`verify_exact_result.py --verify-sources` sees this V2 row-6 provenance, it requires those canonical
auxiliary roles and reruns the certificate-to-header verification.

