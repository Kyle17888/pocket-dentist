# python src/start_vllm.py --config configs/vllm/Qwen3-VL-4B-Instruct.yaml

import os
import sys
import yaml
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Start vLLM server using a YAML config.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--port", type=int, default=None, help="Override the port in the YAML config.")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found.")
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # CLI --port 覆盖 YAML 中的 port
    if args.port is not None:
        config["port"] = args.port

    # GPU assignment: SLURM's allocation takes priority over YAML config.
    # On shared nodes, SLURM assigns different GPUs to different jobs via
    # CUDA_VISIBLE_DEVICES. The YAML config value is only used as a fallback
    # when running outside SLURM (e.g., local development).
    env = os.environ.copy()
    config_gpu = str(config.pop("CUDA_VISIBLE_DEVICES", ""))
    slurm_gpu = os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_STEP_GPUS") or os.environ.get("GPU_DEVICE_ORDINAL")

    if slurm_gpu:
        # SLURM manages GPU assignment — don't override
        print(f"Using SLURM-assigned GPU: CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', 'not set')} (SLURM_JOB_GPUS={slurm_gpu})")
    elif config_gpu:
        # No SLURM — use YAML config value
        env["CUDA_VISIBLE_DEVICES"] = config_gpu
        print(f"Setting environment variable: CUDA_VISIBLE_DEVICES={config_gpu}")

    # Base command
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]

    # Append arguments from YAML
    for key, value in config.items():
        if value is None or value is False:
            continue
        
        # Convert snake_case to kebab-case for vLLM arguments
        arg_name = f"--{key.replace('_', '-')}"
        cmd.append(arg_name)
        
        # If it's True, it's just a boolean flag, don't append "True"
        if value is not True:
            cmd.append(str(value))

    print(f"\nStarting vLLM server with command:\n{' '.join(cmd)}\n")
    
    # Use subprocess.run to replace the Python wrapper process with vLLM
    try:
        subprocess.run(cmd, check=True, env=env)
    except KeyboardInterrupt:
        print("Shutting down vLLM server...")

if __name__ == "__main__":
    main()
