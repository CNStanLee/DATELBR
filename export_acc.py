import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import os
import shutil

def estimate():
    model_dir = os.environ['FINN_ROOT'] + "/scripts/DATELBR/model"
    model_file = model_dir + "/bls_pinv_brevitas_simple_caseA1_cyclehalf_wb4_ab4.onnx"

    estimates_output_dir = "output_estimates_only"

    #Delete previous run results if exist
    if os.path.exists(estimates_output_dir):
        shutil.rmtree(estimates_output_dir)
        print("Previous run results deleted!")


    cfg_estimates = build.DataflowBuildConfig(
        output_dir          = estimates_output_dir,
        mvau_wwidth_max     = 80,
        target_fps          = 1000000,
        synth_clk_period_ns = 10.0,
        fpga_part           = "xc7z020clg400-1",
        steps               = build_cfg.estimate_only_dataflow_steps,
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        ]
    )

    build.build_dataflow_cfg(model_file, cfg_estimates)

def export_dense_acc(brevitas_model,
               output_dir="output_acc",
               mvau_wwidth_max=80,
               #target_fps=1000000,
               synth_clk_period_ns=2.0,
               board="ZCU111",
               steps=build_cfg.estimate_only_dataflow_steps):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print("Previous run results deleted!")

    cfg_deployment = build.DataflowBuildConfig(
        output_dir          = output_dir,
        mvau_wwidth_max     = mvau_wwidth_max,
        #target_fps          = target_fps,
        synth_clk_period_ns = synth_clk_period_ns,
        board               = board,
        steps               = steps,
        specialize_layers_config_file = os.environ['FINN_ROOT'] + "/scripts/DATELBR/config/impl_style.json",
        folding_config_file = os.environ['FINN_ROOT'] + "/scripts/DATELBR/config/folding_config.json",
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.STITCHED_IP,
            build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
            build_cfg.DataflowOutputType.OOC_SYNTH,
        ]
    )
    build.build_dataflow_cfg(brevitas_model, cfg_deployment)

def main():
    print("start gernerating accs...")
    # estimate()
    
    # dense acc

    # export_dense_acc(
    #     brevitas_model=os.environ['FINN_ROOT'] + "/scripts/DATELBR/model/bls_pinv_brevitas_simple_caseA1_cyclehalf_wb3_ab3.onnx",
    #     output_dir="output_acc_caseA1_cyclehalf_wb3_ab3",
    #     mvau_wwidth_max=1200,
    #     # target_fps=10000000000000000,
    #     synth_clk_period_ns=2.0,
    #     board="ZCU104",
    #     #steps=build_cfg.estimate_only_dataflow_steps # default_build_dataflow_steps
    #     steps=build_cfg.default_build_dataflow_steps
    # )
    # print("finished gernerating accs.")


    # sparse acc

    export_dense_acc(
        brevitas_model=os.environ['FINN_ROOT'] + "/scripts/DATELBR/model/bls_pinv_brevitas_simple_caseA1_cyclehalf_wb3_ab3_prune0.9.onnx",
        output_dir="output_acc_caseA1_cyclehalf_wb3_ab3_prune",
        mvau_wwidth_max=1200,
        # target_fps=10000000000000000,
        synth_clk_period_ns=2.0,
        board="ZCU104",
        #steps=build_cfg.estimate_only_dataflow_steps # default_build_dataflow_steps
        steps=build_cfg.default_build_dataflow_steps
    )
    print("finished gernerating accs.")

if __name__ == "__main__":
    main()