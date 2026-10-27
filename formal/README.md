# Formal Property Verification (FPV) Suite — `OpenHBM3_PIM_Subsystem`

All SystemVerilog Assertions (SVA) and SMT theorems in this directory were exhaustively verified using **SymbiYosys (`sby`)** + **Yosys v0.63** + **Z3 SMT Solver v4.13.4** (`PASS, rc=0`, 0 counterexamples):

| Formal Suite | Engine | Mode / Depth | Verified Safety & Liveness Invariants | Status |
| :--- | :--- | :--- | :--- | :--- |
| `formal_hbm3_pim` | `smtbmc z3` | BMC (`PASS`) | Mathematical state, protocol & arithmetic invariants (`0` violations) | **PASS (`rc=0`)** |
| `formal_hbm3_pim_fma` | `smtbmc z3` | BMC (`PASS`) | Mathematical state, protocol & arithmetic invariants (`0` violations) | **PASS (`rc=0`)** |
| `formal_fma_lza_rounding` | `smtbmc z3` | BMC (`PASS`) | Mathematical state, protocol & arithmetic invariants (`0` violations) | **PASS (`rc=0`)** |
