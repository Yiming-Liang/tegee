base_cmd = "python3 finetune0.py --config /cpfs/29cd2992fe666f2a/user/zhangge/xw/trainlarge/lr_0.0003/lorabase.json --bank_config /cpfs/29cd2992fe666f2a/user/zhangge/xw/greedysoup/info_dict_7.json --data_fp /cpfs/29cd2992fe666f2a/shared/public/self-ins/default/ --model_path /cpfs/29cd2992fe666f2a/user/zhangge/xw/trainlarge/delta_model_base"

bank_ids = list(range(834))
samples = [5, 50, 100, 500]

# We have 833 bank IDs and 4 sample sizes. So, in total, we have 833 * 4 = 3332 command lines.
# Each script should contain approximately 3332 / 8 = 416 command lines.
commands = []

for bank_id in bank_ids:
    for sample in samples:
        cmd = f"{base_cmd} --bank_ids {bank_id} --samples {sample} --lora_save_dir ./gpt_soup/test_base_{sample}/"
        commands.append(cmd)

commands_per_script = len(commands) // 8

for i in range(8):
    script_filename = f"script_base_part_{i}.sh"
    with open(script_filename, 'w') as script_file:
        script_file.write("#!/bin/bash\n")
        
        start_idx = i * commands_per_script
        end_idx = start_idx + commands_per_script
        for cmd in commands[start_idx:end_idx]:
            script_file.write(f"{cmd}\n")

    print(f"Commands written to {script_filename}")
