import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--type", default='', type=str,
                    help="the datapath of meta files")
parser.add_argument("--bank_config", default='max_length', type=str,
                    help="the datapath of meta files")
parser.add_argument("--output", default='max_length', type=str,
                    help="the datapath of meta files")
parser.add_argument("--output_script", default='max_length', type=str,
                    help="the datapath of meta files")
parser.add_argument("--split", default=8, type=int,
                    help="the datapath of meta files")
args = parser.parse_args()

base_cmd = f"python3 finetune0.py --config /cpfs/29cd2992fe666f2a/user/zhangge/xw/trainlarge/lr_0.0003/lora{args.type}.json --bank_config {args.bank_config} --data_fp /cpfs/29cd2992fe666f2a/shared/public/self-ins/default/ --model_path /cpfs/29cd2992fe666f2a/user/zhangge/xw/trainlarge/delta_model_{args.type}"

bank_ids = list(range(819))
samples = [5, 50, 100, 500]

# We have 833 bank IDs and 4 sample sizes. So, in total, we have 833 * 4 = 3332 command lines.
# Each script should contain approximately 3332 / 8 = 416 command lines.
commands = []

for bank_id in bank_ids:
    for sample in samples:
        cmd = f"{base_cmd} --bank_ids {bank_id} --samples {sample} --lora_save_dir ./{args.output}/test_{args.type}_{sample}/"
        commands.append(cmd)

commands_per_script = len(commands) // args.split 

for i in range(args.split):
    script_filename = f"script_generator_{args.type}_part_{i}.sh"
    with open(script_filename, 'w') as script_file:
        script_file.write("#!/bin/bash\n")
        
        start_idx = i * commands_per_script
        end_idx = start_idx + commands_per_script
        for cmd in commands[start_idx:end_idx]:
            script_file.write(f"{cmd}\n")

    print(f"Commands written to {script_filename}")
