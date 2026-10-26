# OpenHBM3_PIM_Subsystem: Hardware Integration & Pinout Guide
**Base-Die Processing-in-Memory System Architecture (AXI5 & MMIO Protocols)**

---

## 1. Top-Level Module Instantiation

The top-level base-die macro `ara_openhbm3_soc_top` instantiates the decoupled PIM execution engine, the point-to-point interposer crossbar, and the dual-stack HBM3 memory channels.

### SystemVerilog Instantiation Template

```systemverilog
ara_openhbm3_soc_top #(
    .DATA_WIDTH   (256),   // 256-bit wide internal TSV data bus
    .ADDR_WIDTH   (32),    // 32-bit linear address space
    .NUM_LANES    (16)     // 16 parallel vector execution lanes
) u_hbm3_pim_subsystem (
    // Clock & Asynchronous Reset
    .clk_hbm_i            (clk_pim_1000mhz),     // Base-Die PIM clock (1.0 GHz)
    .rst_async_n          (sys_rst_n),           // Master asynchronous reset

    // Host AXI5 Master 0 Interface (Direct DRAM Access)
    .m0_axi_awaddr        (host_m0_awaddr),
    .m0_axi_awvalid       (host_m0_awvalid),
    .m0_axi_awready       (host_m0_awready),
    .m0_axi_wdata         (host_m0_wdata),
    .m0_axi_wvalid        (host_m0_wvalid),
    .m0_axi_wready        (host_m0_wready),
    .m0_axi_bvalid        (host_m0_bvalid),
    .m0_axi_bready        (host_m0_bready),
    .m0_axi_araddr        (host_m0_araddr),
    .m0_axi_arvalid       (host_m0_arvalid),
    .m0_axi_arready       (host_m0_arready),
    .m0_axi_rdata         (host_m0_rdata),
    .m0_axi_rvalid        (host_m0_rvalid),
    .m0_axi_rready        (host_m0_rready),

    // MMIO PIM Task Dispatcher
    .pim_cmd_valid_i      (host_pim_cmd_valid),  // Strobe to trigger task
    .pim_cmd_ready_o      (host_pim_cmd_ready),  // PIM ready for new task
    .pim_desc_0_i         (host_pim_desc_0),     // [Opcode, Precision, Flags]
    .pim_desc_1_i         (host_pim_desc_1),     // Base Weight Address
    .pim_desc_2_i         (host_pim_desc_2),     // Base Activation Address
    .pim_desc_3_i         (host_pim_desc_3),     // Matrix Dimensions [K_LEN, N_LEN]
    .pim_desc_4_i         (host_pim_desc_4),     // Destination Address
    
    // Status & Fault Traps
    .pim_busy_o           (pim_engine_busy),     // Asserted during compute
    .pim_done_irq_o       (pim_completion_irq),  // Task completion interrupt
    .pim_error_code_o     (pim_fault_code[3:0]), // Security / Bounds fault code
    .pim_force_abort_i    (host_emergency_abort) // Anti-brick 1-cycle abort
);
```

---

## 2. MMIO Command Descriptor Specification

Host processors offload tasks to the PIM base-die via five 32-bit Memory-Mapped I/O (MMIO) registers:

### Register Field Definitions

#### Descriptor 0 (`pim_desc_0_i`): Task Control & Opcode
```
+─────────────────────────────────────────────────────────────────────────────+
|                         DESCRIPTOR 0 BIT ALLOCATION                         |
|                                                                             |
|  [31:24]         [23:20]          [19:16]        [15:8]         [7:0]       |
| Reserved      Activation Mode   Data Format     Stack Mask     Opcode       |
+─────────────────────────────────────────────────────────────────────────────+
```
- **Opcode (`[7:0]`)**:
  * `0x01`: Matrix Multiplication (GEMM Contraction).
  * `0x02`: General Matrix-Vector Multiplication (GEMV).
  * `0x03`: Softmax Exponentiation & Normalization.
  * `0x04`: Root Mean Square Normalization (RMSNorm).
  * `0x05`: Fused SwiGLU Non-Linear Feed-Forward Block.
- **Stack Mask (`[15:8]`)**: Bitmask designating which physical HBM3 stacks participate in parallel.
- **Data Format (`[19:16]`)**:
  * `0x0`: IEEE-754 FP16 (Half Precision).
  * `0x1`: BF16 (Brain Floating Point 16).
  * `0x2`: OCP FP8 E4M3 / E5M2.
  * `0x3`: Signed INT8 / INT16 Bypass.
- **Activation Mode (`[23:20]`)**:
  * `0x0`: None (Linear output).
  * `0x1`: Fused ReLU clamp.
  * `0x2`: Continuous Piecewise Linear (PWL) SiLU / Swish.
  * `0x3`: Fast Reciprocal Square-Root ($1 / \sqrt{x}$).

#### Descriptor 1 (`pim_desc_1_i`): Weight Tensor Base Address
- Physical address in DRAM pointing to weight matrix $W$ ($32$-byte aligned).

#### Descriptor 2 (`pim_desc_2_i`): Activation Input Base Address
- Physical address pointing to activation vector $X$ in local TCDM or DRAM.

#### Descriptor 3 (`pim_desc_3_i`): Matrix Dimensions
- `[15:0]`: Inner contraction dimension length ($K_{LEN}$).
- `[31:16]`: Outer output dimension length ($N_{LEN}$).

#### Descriptor 4 (`pim_desc_4_i`): Output Destination Base Address
- Target address where computed output vector $Y$ is stored.

---

## 3. Voltage Regulator Specifications & Power Supply Network

The base-die hosts high-density digital logic directly under thermal dissipation from the DRAM stacks. Power supply noise must be tightly regulated.

| Power Rail | Nominal Voltage | Tolerance | Maximum Ripple | Transient Response | Function |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`VDD_PIM`** | **0.80 V** | **$\pm 3.0\%$ ($\pm 24\text{ mV}$)** | **$< 12\text{ mV}$** | **$< 15\text{ ns}$ (0 to 12A step)** | 16-Lane True-FMA and reduction trees. |
| **`VDD_DRAM`** | **1.10 V** | **$\pm 3.0\%$ ($\pm 33\text{ mV}$)** | **$< 15\text{ mV}$** | **$< 50\text{ ns}$** | 3D DRAM core array supply. |
| **`VDD_IF`** | **0.75 V** | **$\pm 5.0\%$ ($\pm 37.5\text{ mV}$)** | **$< 15\text{ mV}$** | **$< 25\text{ ns}$** | CoWoS interposer AXI5 bus transceivers. |

### Decoupling & Thermal Packaging
1. **On-Die Integrated Voltage Regulators (IVR)**: Uses integrated multi-phase switched-inductor bucks fabricated directly on the logic base die, converting an incoming $1.8\text{V}$ intermediate rail down to $0.80\text{V}$ with $>88\%$ efficiency.
2. **Deep Trench Capacitance**: $180\text{ nF/mm}^2$ capacitance distributed uniformly under the arithmetic units to absorb fast di/dt steps during matrix dispatch.

---

## 4. Safety Guard Traps & Anti-Brick Protection

The PIM sub-system includes hardware-level traps to protect system stability:

1. **Anti-Brick Software Abort (`pim_force_abort_i`)**:
   - If a host driver thread dies or software hangs, asserting `pim_force_abort_i` terminates all active PIM pipelines within **1 clock cycle**, clears all internal FIFO queues, resets FSMs to `IDLE`, and deasserts `pim_busy_o`.
2. **Out-of-Bounds Memory Security Guard**:
   - If any descriptor requests addresses beyond valid DRAM boundaries (e.g. $\ge \text{0xF000\_0000}$), execution is halted immediately without modifying memory, setting `pim_error_code_o = 0x5` and raising an exception interrupt.
3. **Zero-Dimension Security Trap**:
   - Supplying $K_{LEN} = 0$ or $N_{LEN} = 0$ triggers an immediate trap with `pim_error_code_o = 0x6`, preventing infinite hardware divide-by-zero loops.
