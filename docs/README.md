# Documentation

Technical documentation of automatic-llvm-mca. The top-level [README](../README.md) covers building
and running the tools.

| Document | Contents |
|---|---|
| [docs.md](docs.md) | Technical overview: scheduling-model customizations for the Neoverse-derived models and the Cortex-X1 (section 1), the MCA steady-state machinery (section 2), the MLP calculation (section 3), the SOG-derived notes of the Facile predictor for the out-of-order cores (section 4), and the scheduling-model plumbing (section 5). |
| [a55-sog.md](a55-sog.md) | The Cortex-A55 model (TableGen model, issue stage, bypasses, corrections) and its conformance to the Arm Cortex-A55 Software Optimization Guide, item by item (compatible / not compatible / deliberate deviations / unclear / inconsistent in the guide). Written by Claude (Anthropic) on 2026-10-31; check against the guide. |
| [a520-sog.md](a520-sog.md) | The Cortex-A520 model (TableGen model and its in-order issue stage) and its conformance to the Arm Cortex-A520 Core Software Optimization Guide, item by item (compatible / not compatible / deliberate deviations / unclear / inconsistent in the guide / outside the A520 path / borrowed from the Cortex-A55, not SOG-based). Written by Claude (Anthropic) on 2026-10-31; check against the guide. |
| [a76-sog.md](a76-sog.md) | The Cortex-A76 model (`NeoverseN1Model` in `AArch64SchedNeoverseN1.td`, installed only for cortex-a76/-a76ae, and the Facile paths that predict it) and its conformance to the Arm Cortex-A76 Software Optimization Guide, item by item (compatible / not compatible / deliberate deviations / unclear / inconsistent in the guide / outside the A76 path / taken over from the in-order cores, not SOG-based). Written by Claude (Anthropic) on 2026-10-31; check against the guide. |
| [a78-sog.md](a78-sog.md) | The Cortex-A78 model (`NeoverseN2Model` in `AArch64SchedNeoverseN2.td`, installed only for cortex-a78/-a78ae/-a78c, and the Facile paths that predict it: fusion table, zero-latency MOVs) and its conformance to the Arm Cortex-A78 Core Software Optimization Guide, item by item (compatible / not compatible / deliberate deviations / unclear / inconsistent in the guide / outside the A78 path / taken over from other cores, not SOG-based). Written by Claude (Anthropic) on 2026-10-31; check against the guide. |

The Arm Software Optimization Guides themselves are not part of the repository (the working copies
in `tmp/` are not tracked).
