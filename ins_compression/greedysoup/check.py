import os
import json

base_path = '.'  # 从当前目录开始
merged_data = {}

# 遍历base_path下的所有子目录
def merge_miss():
    for subdir in os.listdir(base_path):
        if subdir[-4:]=='soup':
            subdir_path = os.path.join(base_path, subdir)
            # 只检查目录
            if os.path.isdir(subdir_path):
                output_file = os.path.join(subdir_path, "output.json")
                # 检查该目录下是否有output.json文件
                if os.path.exists(output_file):
                    with open(output_file, 'r') as f:
                        # 加载json数据并将其合并到merged_data中
                        data = json.load(f)
                        merged_data[subdir] = data

    # 将合并后的数据保存为新的json文件
    with open('merged_output.json', 'w') as f:
        json.dump(merged_data, f, indent=4)

    print("Merged data saved to merged_output.json")

base_path = "/cpfs/29cd2992fe666f2a/shared/public/self-ins/default/task"
save_dir_base = "delta_model_base/task"
config_path = "/cpfs/29cd2992fe666f2a/user/zhangge/xw/trainlarge/lr_0.0003/lora"
config_bank = "/cpfs/29cd2992fe666f2a/user/zhangge/xw/greedysoup/info_dict_"
data_fps = '/cpfs/29cd2992fe666f2a/shared/public/self-ins/default/'
model_path = '/cpfs/29cd2992fe666f2a/user/zhangge/xw/trainlarge/delta_model_'
lora_path = '/cpfs/29cd2992fe666f2a/user/zhangge/xw/greedysoup/'

def allocate_task():
    with open('/cpfs/29cd2992fe666f2a/user/zhangge/xw/greedysoup/merged_output.json', 'r') as file:
        data = json.load(file)
    tasks = []
    # 遍历合并后的数据中的每个子目录
    for subdir, subdir_data in data.items():
        # 遍历每个子目录中的文件数据
        for folders,missing_folders in subdir_data.items():
            for item in missing_folders:
                sample = folders.split('_')[2]
                degree = folders.split('_')[1]
                types = subdir.split('_')[0]
                tasks.append([item,degree,sample,types])
    N = len(tasks)
    gpus = 8
    gpu_dic = {}
    # tasks_per_gpu = N // gpus
    task_count = 0
    for task in tasks:
        gpu_id = task_count % 8
        task_count+=1
        command = f"python3 finetune0.py --config {config_path + task[1]+'.json'} --bank_config {config_bank + task[3]+'_7.json'} --data_fp {data_fps} --model_path {model_path+task[1]} --bank_ids {str(task[0])}  --samples {task[2]} --lora_save_dir {lora_path}{task[3]+'_soup/'}{'test_'+task[1]+'_'+task[2]+'/'}"
        gpu_dic.setdefault(gpu_id, []).append(command)

    for gpu_id in range(gpus):
        with open(f"/cpfs/29cd2992fe666f2a/user/zhangge/xw/greedysoup/script_miss_part_{gpu_id}.sh", "w") as file:
            file.write("#!/bin/bash\n")
            for command in gpu_dic[gpu_id]:
                file.write(command + "\n")

    print(f"Scripts saved from s_0.sh to s_{gpus-1}.sh")




# merge_miss()
allocate_task()