//Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
//--------------------------------------------------------------------------------
//Tool Version: Vivado v.2022.2 (lin64) Build 3671981 Fri Oct 14 04:59:54 MDT 2022
//Date        : Sun Nov 30 06:47:19 2025
//Host        : finn_dev_changhong running 64-bit Ubuntu 22.04.1 LTS
//Command     : generate_target finn_design.bd
//Design      : finn_design
//Purpose     : IP block netlist
//--------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

module MVAU_rtl_0_imp_1DNJB9Y
   (ap_clk,
    ap_rst_n,
    in0_V_tdata,
    in0_V_tready,
    in0_V_tvalid,
    out0_V_tdata,
    out0_V_tready,
    out0_V_tvalid);
  input ap_clk;
  input ap_rst_n;
  input [23:0]in0_V_tdata;
  output in0_V_tready;
  input in0_V_tvalid;
  output [31:0]out0_V_tdata;
  input out0_V_tready;
  output out0_V_tvalid;

  wire [31:0]MVAU_rtl_0_out0_V_TDATA;
  wire MVAU_rtl_0_out0_V_TREADY;
  wire MVAU_rtl_0_out0_V_TVALID;
  wire [63:0]MVAU_rtl_0_wstrm_m_axis_0_TDATA;
  wire MVAU_rtl_0_wstrm_m_axis_0_TREADY;
  wire MVAU_rtl_0_wstrm_m_axis_0_TVALID;
  wire ap_clk_1;
  wire ap_rst_n_1;
  wire [23:0]in0_V_1_TDATA;
  wire in0_V_1_TREADY;
  wire in0_V_1_TVALID;

  assign MVAU_rtl_0_out0_V_TREADY = out0_V_tready;
  assign ap_clk_1 = ap_clk;
  assign ap_rst_n_1 = ap_rst_n;
  assign in0_V_1_TDATA = in0_V_tdata[23:0];
  assign in0_V_1_TVALID = in0_V_tvalid;
  assign in0_V_tready = in0_V_1_TREADY;
  assign out0_V_tdata[31:0] = MVAU_rtl_0_out0_V_TDATA;
  assign out0_V_tvalid = MVAU_rtl_0_out0_V_TVALID;
  finn_design_MVAU_rtl_0_0 MVAU_rtl_0
       (.ap_clk(ap_clk_1),
        .ap_clk2x(ap_clk_1),
        .ap_rst_n(ap_rst_n_1),
        .in0_V_TDATA(in0_V_1_TDATA),
        .in0_V_TREADY(in0_V_1_TREADY),
        .in0_V_TVALID(in0_V_1_TVALID),
        .in1_V_TDATA(MVAU_rtl_0_wstrm_m_axis_0_TDATA),
        .in1_V_TREADY(MVAU_rtl_0_wstrm_m_axis_0_TREADY),
        .in1_V_TVALID(MVAU_rtl_0_wstrm_m_axis_0_TVALID),
        .out0_V_TDATA(MVAU_rtl_0_out0_V_TDATA),
        .out0_V_TREADY(MVAU_rtl_0_out0_V_TREADY),
        .out0_V_TVALID(MVAU_rtl_0_out0_V_TVALID));
  finn_design_MVAU_rtl_0_wstrm_0 MVAU_rtl_0_wstrm
       (.ap_clk(ap_clk_1),
        .ap_clk2x(ap_clk_1),
        .ap_rst_n(ap_rst_n_1),
        .m_axis_0_tdata(MVAU_rtl_0_wstrm_m_axis_0_TDATA),
        .m_axis_0_tready(MVAU_rtl_0_wstrm_m_axis_0_TREADY),
        .m_axis_0_tvalid(MVAU_rtl_0_wstrm_m_axis_0_TVALID),
        .s_axilite_ARADDR({1'b0,1'b0,1'b0}),
        .s_axilite_ARPROT({1'b0,1'b0,1'b0}),
        .s_axilite_ARVALID(1'b0),
        .s_axilite_AWADDR({1'b0,1'b0,1'b0}),
        .s_axilite_AWPROT({1'b0,1'b0,1'b0}),
        .s_axilite_AWVALID(1'b0),
        .s_axilite_BREADY(1'b0),
        .s_axilite_RREADY(1'b0),
        .s_axilite_WDATA({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .s_axilite_WSTRB({1'b1,1'b1,1'b1,1'b1}),
        .s_axilite_WVALID(1'b0),
        .s_axis_0_tdata(1'b0),
        .s_axis_0_tvalid(1'b0));
endmodule

(* CORE_GENERATION_INFO = "finn_design,IP_Integrator,{x_ipVendor=xilinx.com,x_ipLibrary=BlockDiagram,x_ipName=finn_design,x_ipVersion=1.00.a,x_ipLanguage=VERILOG,numBlks=11,numReposBlks=10,numNonXlnxBlks=0,numHierBlks=1,maxHierDepth=1,numSysgenBlks=0,numHlsBlks=2,numHdlrefBlks=8,numPkgbdBlks=0,bdsource=USER,synth_mode=OOC_per_IP}" *) (* HW_HANDOFF = "finn_design.hwdef" *) 
module finn_design
   (ap_clk,
    ap_rst_n,
    m_axis_0_tdata,
    m_axis_0_tready,
    m_axis_0_tvalid,
    s_axis_0_tdata,
    s_axis_0_tready,
    s_axis_0_tvalid);
  (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 CLK.AP_CLK CLK" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME CLK.AP_CLK, ASSOCIATED_BUSIF s_axis_0:m_axis_0, ASSOCIATED_RESET ap_rst_n, CLK_DOMAIN finn_design_ap_clk_0, FREQ_HZ 500000000, FREQ_TOLERANCE_HZ 0, INSERT_VIP 0, PHASE 0.0" *) input ap_clk;
  (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 RST.AP_RST_N RST" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME RST.AP_RST_N, INSERT_VIP 0, POLARITY ACTIVE_LOW" *) input ap_rst_n;
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis_0 " *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME m_axis_0, CLK_DOMAIN finn_design_ap_clk_0, FREQ_HZ 500000000, HAS_TKEEP 0, HAS_TLAST 0, HAS_TREADY 1, HAS_TSTRB 0, INSERT_VIP 0, LAYERED_METADATA undef, PHASE 0.0, TDATA_NUM_BYTES 4, TDEST_WIDTH 0, TID_WIDTH 0, TUSER_WIDTH 0" *) output [31:0]m_axis_0_tdata;
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis_0 " *) input m_axis_0_tready;
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis_0 " *) output m_axis_0_tvalid;
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis_0 " *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME s_axis_0, CLK_DOMAIN finn_design_ap_clk_0, FREQ_HZ 500000000, HAS_TKEEP 0, HAS_TLAST 0, HAS_TREADY 1, HAS_TSTRB 0, INSERT_VIP 0, LAYERED_METADATA undef, PHASE 0.0, TDATA_NUM_BYTES 128, TDEST_WIDTH 0, TID_WIDTH 0, TUSER_WIDTH 0" *) input [1023:0]s_axis_0_tdata;
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis_0 " *) output s_axis_0_tready;
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis_0 " *) input s_axis_0_tvalid;

  wire [191:0]MVAU_hls_0_out0_V_TDATA;
  wire MVAU_hls_0_out0_V_TREADY;
  wire MVAU_hls_0_out0_V_TVALID;
  wire [23:0]MVAU_hls_1_out0_V_TDATA;
  wire MVAU_hls_1_out0_V_TREADY;
  wire MVAU_hls_1_out0_V_TVALID;
  wire [31:0]MVAU_rtl_0_out0_V_TDATA;
  wire MVAU_rtl_0_out0_V_TREADY;
  wire MVAU_rtl_0_out0_V_TVALID;
  wire [1023:0]StreamingFIFO_rtl_0_out0_V_TDATA;
  wire StreamingFIFO_rtl_0_out0_V_TREADY;
  wire StreamingFIFO_rtl_0_out0_V_TVALID;
  wire [95:0]StreamingFIFO_rtl_1_out0_V_TDATA;
  wire StreamingFIFO_rtl_1_out0_V_TREADY;
  wire StreamingFIFO_rtl_1_out0_V_TVALID;
  wire [191:0]StreamingFIFO_rtl_2_out0_V_TDATA;
  wire StreamingFIFO_rtl_2_out0_V_TREADY;
  wire StreamingFIFO_rtl_2_out0_V_TVALID;
  wire [23:0]StreamingFIFO_rtl_3_out0_V_TDATA;
  wire StreamingFIFO_rtl_3_out0_V_TREADY;
  wire StreamingFIFO_rtl_3_out0_V_TVALID;
  wire [31:0]StreamingFIFO_rtl_4_out0_V_TDATA;
  wire StreamingFIFO_rtl_4_out0_V_TREADY;
  wire StreamingFIFO_rtl_4_out0_V_TVALID;
  wire [95:0]Thresholding_rtl_0_out0_V_TDATA;
  wire Thresholding_rtl_0_out0_V_TREADY;
  wire Thresholding_rtl_0_out0_V_TVALID;
  wire ap_clk_0_1;
  wire ap_rst_n_0_1;
  wire [1023:0]in0_V_0_1_TDATA;
  wire in0_V_0_1_TREADY;
  wire in0_V_0_1_TVALID;

  assign StreamingFIFO_rtl_4_out0_V_TREADY = m_axis_0_tready;
  assign ap_clk_0_1 = ap_clk;
  assign ap_rst_n_0_1 = ap_rst_n;
  assign in0_V_0_1_TDATA = s_axis_0_tdata[1023:0];
  assign in0_V_0_1_TVALID = s_axis_0_tvalid;
  assign m_axis_0_tdata[31:0] = StreamingFIFO_rtl_4_out0_V_TDATA;
  assign m_axis_0_tvalid = StreamingFIFO_rtl_4_out0_V_TVALID;
  assign s_axis_0_tready = in0_V_0_1_TREADY;
  finn_design_MVAU_hls_0_0 MVAU_hls_0
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_TDATA(StreamingFIFO_rtl_1_out0_V_TDATA),
        .in0_V_TREADY(StreamingFIFO_rtl_1_out0_V_TREADY),
        .in0_V_TVALID(StreamingFIFO_rtl_1_out0_V_TVALID),
        .out0_V_TDATA(MVAU_hls_0_out0_V_TDATA),
        .out0_V_TREADY(MVAU_hls_0_out0_V_TREADY),
        .out0_V_TVALID(MVAU_hls_0_out0_V_TVALID));
  finn_design_MVAU_hls_1_0 MVAU_hls_1
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_TDATA(StreamingFIFO_rtl_2_out0_V_TDATA),
        .in0_V_TREADY(StreamingFIFO_rtl_2_out0_V_TREADY),
        .in0_V_TVALID(StreamingFIFO_rtl_2_out0_V_TVALID),
        .out0_V_TDATA(MVAU_hls_1_out0_V_TDATA),
        .out0_V_TREADY(MVAU_hls_1_out0_V_TREADY),
        .out0_V_TVALID(MVAU_hls_1_out0_V_TVALID));
  MVAU_rtl_0_imp_1DNJB9Y MVAU_rtl_0
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_tdata(StreamingFIFO_rtl_3_out0_V_TDATA),
        .in0_V_tready(StreamingFIFO_rtl_3_out0_V_TREADY),
        .in0_V_tvalid(StreamingFIFO_rtl_3_out0_V_TVALID),
        .out0_V_tdata(MVAU_rtl_0_out0_V_TDATA),
        .out0_V_tready(MVAU_rtl_0_out0_V_TREADY),
        .out0_V_tvalid(MVAU_rtl_0_out0_V_TVALID));
  finn_design_StreamingFIFO_rtl_0_0 StreamingFIFO_rtl_0
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_TDATA(in0_V_0_1_TDATA),
        .in0_V_TREADY(in0_V_0_1_TREADY),
        .in0_V_TVALID(in0_V_0_1_TVALID),
        .out0_V_TDATA(StreamingFIFO_rtl_0_out0_V_TDATA),
        .out0_V_TREADY(StreamingFIFO_rtl_0_out0_V_TREADY),
        .out0_V_TVALID(StreamingFIFO_rtl_0_out0_V_TVALID));
  finn_design_StreamingFIFO_rtl_1_0 StreamingFIFO_rtl_1
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_TDATA(Thresholding_rtl_0_out0_V_TDATA),
        .in0_V_TREADY(Thresholding_rtl_0_out0_V_TREADY),
        .in0_V_TVALID(Thresholding_rtl_0_out0_V_TVALID),
        .out0_V_TDATA(StreamingFIFO_rtl_1_out0_V_TDATA),
        .out0_V_TREADY(StreamingFIFO_rtl_1_out0_V_TREADY),
        .out0_V_TVALID(StreamingFIFO_rtl_1_out0_V_TVALID));
  finn_design_StreamingFIFO_rtl_2_0 StreamingFIFO_rtl_2
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_TDATA(MVAU_hls_0_out0_V_TDATA),
        .in0_V_TREADY(MVAU_hls_0_out0_V_TREADY),
        .in0_V_TVALID(MVAU_hls_0_out0_V_TVALID),
        .out0_V_TDATA(StreamingFIFO_rtl_2_out0_V_TDATA),
        .out0_V_TREADY(StreamingFIFO_rtl_2_out0_V_TREADY),
        .out0_V_TVALID(StreamingFIFO_rtl_2_out0_V_TVALID));
  finn_design_StreamingFIFO_rtl_3_0 StreamingFIFO_rtl_3
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_TDATA(MVAU_hls_1_out0_V_TDATA),
        .in0_V_TREADY(MVAU_hls_1_out0_V_TREADY),
        .in0_V_TVALID(MVAU_hls_1_out0_V_TVALID),
        .out0_V_TDATA(StreamingFIFO_rtl_3_out0_V_TDATA),
        .out0_V_TREADY(StreamingFIFO_rtl_3_out0_V_TREADY),
        .out0_V_TVALID(StreamingFIFO_rtl_3_out0_V_TVALID));
  finn_design_StreamingFIFO_rtl_4_0 StreamingFIFO_rtl_4
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_TDATA(MVAU_rtl_0_out0_V_TDATA),
        .in0_V_TREADY(MVAU_rtl_0_out0_V_TREADY),
        .in0_V_TVALID(MVAU_rtl_0_out0_V_TVALID),
        .out0_V_TDATA(StreamingFIFO_rtl_4_out0_V_TDATA),
        .out0_V_TREADY(StreamingFIFO_rtl_4_out0_V_TREADY),
        .out0_V_TVALID(StreamingFIFO_rtl_4_out0_V_TVALID));
  finn_design_Thresholding_rtl_0_0 Thresholding_rtl_0
       (.ap_clk(ap_clk_0_1),
        .ap_rst_n(ap_rst_n_0_1),
        .in0_V_TDATA(StreamingFIFO_rtl_0_out0_V_TDATA),
        .in0_V_TREADY(StreamingFIFO_rtl_0_out0_V_TREADY),
        .in0_V_TVALID(StreamingFIFO_rtl_0_out0_V_TVALID),
        .in1_V_TDATA({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .in1_V_TVALID(1'b0),
        .out0_V_TDATA(Thresholding_rtl_0_out0_V_TDATA),
        .out0_V_TREADY(Thresholding_rtl_0_out0_V_TREADY),
        .out0_V_TVALID(Thresholding_rtl_0_out0_V_TVALID),
        .s_axilite_ARADDR({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .s_axilite_ARVALID(1'b0),
        .s_axilite_AWADDR({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .s_axilite_AWVALID(1'b0),
        .s_axilite_BREADY(1'b0),
        .s_axilite_RREADY(1'b0),
        .s_axilite_WDATA({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .s_axilite_WSTRB({1'b1,1'b1,1'b1,1'b1}),
        .s_axilite_WVALID(1'b0));
endmodule
