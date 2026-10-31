# Documentation

Technical documentation of automatic-llvm-mca. The top-level [README](../README.md) covers building
and running the tools.

| Document | Contents |
|---|---|
| [docs.md](docs.md) | Technical overview: scheduling-model customizations for the Neoverse-derived models and the Cortex-X1 (section 1), the MCA steady-state machinery (section 2), the MLP calculation (section 3), the SOG-derived notes of the Facile predictor for the out-of-order cores (section 4), and the scheduling-model plumbing (section 5). |
| [a55-sog.md](a55-sog.md) | The Cortex-A55 model (TableGen model, issue stage, bypasses, corrections) and its conformance to the Arm Cortex-A55 Software Optimization Guide, item by item (compatible / not compatible / deliberate deviations / unclear / inconsistent in the guide). Written by Claude (Anthropic) on 2026-10-31; check against the guide. |

The Arm Software Optimization Guides themselves are not part of the repository (the working copies
in `tmp/` are not tracked).
