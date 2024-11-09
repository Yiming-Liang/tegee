import os
from dataclasses import field, dataclass
from typing import Optional, Any
from transformers import TrainingArguments

from peft import PeftModel
import torch
import transformers
from transformers import Trainer
import functools
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_metric
import logging
import numpy as np

logger = logging.getLogger(__name__)

model_args = None
# from dataset import Seq2SeqDataset, Seq2SeqCollator
def list_to_string(example):
    '''
    convert text to string
    '''
    if isinstance(example, list) :
        seqs = example
        out_str = ''
        for seq in seqs:
            out_str = out_str + seq + '\n'
        return out_str
    else:
        return example   
max_target_length = 512
max_source_length = 512
padding = 'max_length'
ignore_pad_token_for_loss = True

from transformers import (
    AutoModel,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    MBartTokenizer,
    default_data_collator,
    set_seed,
    T5ForConditionalGeneration
)

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from typing import List
import logging
import argparse
logging.basicConfig(level=logging.INFO)

# parser = argparse.ArgumentParser()
# parser.add_argument("--data_fp", default='', type=str,
#                     help="the datapath of meta files")
# parser.add_argument("--padding", default='max_length', type=str,
#                     help="the datapath of meta files")
# parser.add_argument("--do_train", default=True, type=bool,
#                     help="do train")
# parser.add_argument("--max_source_length", default=512, type=int,
#                     help="length of input sentence")
# parser.add_argument("--max_target_length", default=512, type=int,
#                     help="length of output sentence")
# parser.add_argument("--do_val", default=True, type=bool,
#                     help="do val during training")
# parser.add_argument("--do_test", default=True, type=bool,
#                     help="test after training")
# parser.add_argument("--ignore_pad_token_for_loss", default=True, type=bool,
#                     help="")
# parser.add_argument("--config", default='/huangwhai52/gezhang/models/5.5/trainlarge/lr_0.0003/lorabase.json', type=str)
# parser.add_argument("--output_dir", default="outputs/lora/t5-base/", type=str)
# parser.add_argument("--lora_save_dir", default="/huangwhai52/gezhang/models/n_logs/greedysoup/t5-base/5/delta_save/0", type=str)
# parser.add_argument("--loraconfig", default={}, type=dict)
# parser.add_argument("--model_path", default="", type=str)
# parser.add_argument("--model_list", nargs = '+')
# parser.add_argument("--bank_config", default = "/huangwhai52/gezhang/models/5.5/greedysoup_large/5/info_dict.json", type=str)
# parser.add_argument("--bank_ids", default = 0, type=int)
# parser.add_argument("--samples", default = 5, type=int)
# args = parser.parse_args()

class RemainArgHfArgumentParser(HfArgumentParser):
    def parse_json_file(self, json_file: str, return_remaining_args=True ):
        """
        Alternative helper method that does not use `argparse` at all, instead loading a json file and populating the
        dataclass types.
        """
        import argparse
        import json
        from pathlib import Path
        import dataclasses
        data = json.loads(Path(json_file).read_text())
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            inputs = {k: data.pop(k) for k in list(data.keys()) if k in keys}
            obj = dtype(**inputs)
            outputs.append(obj)

        remain_args = argparse.ArgumentParser()
        remain_args.__dict__.update(data)
        if return_remaining_args:
            return (*outputs, remain_args)
        else:
            return (*outputs,)
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            inputs = {k: data.pop(k) for k in list(data.keys()) if k in keys}
            obj = dtype(**inputs)
            outputs.append(obj)

        remain_args = argparse.ArgumentParser()
        remain_args.__dict__.update(data)
        if return_remaining_args:
            return (*outputs, remain_args)
        else:
            return (*outputs,)

@dataclass
class CustomArguments:
    info_dict: str = field(default='', metadata={"help": "info dictionary "})
    custom_def: str = field(default='', metadata={"help": "custom defination path"})
    lora_path: str = field(default='', metadata={"help": "lora path"})
    samples: int = field(default=5, metadata={"help": "Another custom argument"})
    bank_ids: int = field(default=10, metadata={"help": "Another custom argument"})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_name_or_path: Optional[str] = field(default="")
    data_paths: List[str] = field(default_factory=lambda: [''], metadata={"help": "Path to the training data."})
    instruction_length: int = 40
    output_length: int = 160
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    load_in_8bit: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    # lora arguments
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q", "v",])
    # bank_config: Optional[str] = field(default=None)
    # bank_ids: Optional[int] = field(default=7)

class get_data():
    '''
    load data model
    '''
    def __init__(self,args, taskname, samples, concluded_instructions):
        self.taskname = taskname 
        self.do_train = args.do_train
        self.do_val = args.do_eval
        self.do_test = args.do_predict
        self.samples = samples
        self.concluded_instructiosn = concluded_instructions

        self.addins = True 

 

    def make(self):
        # print(self.taskname)
        with open(self.taskname, encoding='UTF-8') as f:
            task = json.load(f)
        if len(task['Definition']) > 1:
            merged = " ".join([string for string in task['Definition'] if string])
            task['Definition'] = [merged]
        ins = task['Instances']
        length = len(ins)
        df = pd.DataFrame(ins)
        df['instructions'] = task['Definition']*length
        if self.addins:     #add instructions
            df['source'] = self.concluded_instructiosn + '\n'  + df['input']
            df['target'] = df['output'].map(list_to_string)
        else:
            df['source'] = df['input']
            df['target'] = df['output'].map(list_to_string)
        trainval_dt, test_dt = train_test_split(df, random_state=42, train_size=0.9)
        train_dt , val_dt = train_test_split(trainval_dt, random_state=42, train_size=0.9)
        if len(train_dt)>=self.samples:
            train_dt = train_dt[:self.samples]
        return train_dt, test_dt, val_dt


def train():
    parser = RemainArgHfArgumentParser((TrainingArguments, CustomArguments))
    model_args, custom_args = parser.parse_args_into_dataclasses()
    model_name_or_path = model_args.model_name_or_path

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        model_max_length=model_args.model_max_length,
        padding_side="right",
        use_fast=False
    )

    device_map = "auto"

    if model_args.model_name_or_path == "google/flan-t5-xxl" and model_args.load_in_8bit == False:
        logging.info("You are training flan-t5-xxl with float32 data type. "
                     "To save the memory, you may set load_in_8bit to True.")


    set_seed(model_args.seed)
    #model average
    model_paths = []
    with open(custom_args.info_dict) as fp:
        bank = json.load(fp)
    data_fp = f"{model_args.data_paths[0]}/{bank[str(custom_args.bank_ids)][0]}"
    model_list = bank[str(custom_args.bank_ids)][1]
    model_similarity_list = bank[str(custom_args.bank_ids)][2]
    for md in model_list: #initialize a model
        model_paths.append(f"{custom_args.lora_path}/{md[:-5]}")
    print(model_paths, data_fp)
    # exit()
    NUM_MODELS = len(model_paths)
    for j, model_path in enumerate(model_paths):
        state_dict = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            load_in_8bit=model_args.load_in_8bit,
            use_cache=False,
            torch_dtype=torch.float16,
            cache_dir=model_args.cache_dir,
            ignore_mismatched_sizes=True
        )
        peft_model = PeftModel.from_pretrained(state_dict, model_path)


        state_dict = peft_model.state_dict()
        if j == 0:
            uniform_soup = {k : v * model_similarity_list[j] for k, v in state_dict.items()}
        else:
            uniform_soup = {k : v * model_similarity_list[j] + uniform_soup[k] for k, v in state_dict.items()}
    
        # if j == 0:
        #     uniform_soup = {k : v * (1./NUM_MODELS) for k, v in state_dict.items()}
        # else:
        #     uniform_soup = {k : v * (1./NUM_MODELS) + uniform_soup[k] for k, v in state_dict.items()}
    
    # model = AutoModelForSeq2SeqLM.from_pretrained(
    #     model_args.model_name_or_path,
    #     from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #     config=config,
    #     cache_dir=model_args.cache_dir,
    #     revision=model_args.model_revision,
    #     use_auth_token=True if model_args.use_auth_token else None,
    #     ignore_mismatched_sizes=True
    # )
    for param_tensor in uniform_soup:
        print(param_tensor, "\t", uniform_soup[param_tensor].size())
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        load_in_8bit=model_args.load_in_8bit,
        use_cache=False,
        torch_dtype=torch.float16,
        cache_dir=model_args.cache_dir,
        device_map=device_map,
    )
    config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules=model_args.lora_target_modules,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, config)
    print(' --- ' * 20)
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    model.load_state_dict(uniform_soup)
    # model.print_trainable_parameters()
    # data_fp = '{}/task{}.json'.format(bank_ids) 
    # print(data_fp)
    custom_definations = {}
    with open(custom_args.custom_def, 'r') as f:
        # custom_definations = json.load(f)
        for line in f:
            task_def = json.loads(line.strip()) 
            custom_definations[task_def['task']] = task_def['instruction']
    # print(custom_definations)
    custom_def = custom_definations[bank[str(custom_args.bank_ids)][0]] 
    dt = get_data(model_args, data_fp, custom_args.samples, custom_def)    #get data
    train_dataset, test_dataset, val_dataset = dt.make()
    train_datasets = Dataset.from_pandas(train_dataset)
    print(train_datasets[0]['source'])
    print(train_datasets[0]['target'])
    test_datasets = Dataset.from_pandas(test_dataset)
    # print(test_datasets[0]['source'])
    # print(test_datasets[0]['target'])
    # exit()
    eval_datasets = Dataset.from_pandas(val_dataset)
    column_names = ['source', 'target','__index_level_0__']


    def preprocess_function(examples):
        '''
        inheritate from opendelta example
        '''
        ignore_pad_token_for_loss = True
        model_inputs = tokenizer([s for s in examples['source']], max_length=max_source_length,
                                 padding=padding, truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer([t for t in examples['target']], max_length=max_target_length, padding=padding, truncation=True)
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = train_datasets.map(
                functools.partial(preprocess_function),
                batched=True,
                remove_columns=column_names, 
            )

    eval_dataset = eval_datasets.map(
                functools.partial(preprocess_function),
                batched=True,
                remove_columns=column_names, 
            )
    
    test_dataset = test_datasets.map(
                functools.partial(preprocess_function),
                batched=True,
                remove_columns=column_names
            )

    # Data collator
    ignore_pad_token_for_loss = True
    label_pad_token_id = -100 if ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = default_data_collator

    # dataset = Seq2SeqDataset(args.data_paths)
    # collator = Seq2SeqCollator(tokenizer, args.instruction_length, args.output_length)

    trainer = Trainer(
        model,
        args=model_args,
        data_collator=data_collator,
        train_dataset=train_dataset if model_args.do_train else None,
        eval_dataset=eval_dataset if model_args.do_eval else None,
        tokenizer=tokenizer,
        # compute_metrics=compute_metrics
        # train_dataset=train_dataset,
    )
    # trainer = Seq2SeqTrainer(
    #     model=model,
    #     args=args,
    #     # delta_args=delta_args,
    #     train_dataset=train_dataset if args.do_train else None,
    #     eval_dataset=eval_dataset if args.do_eval else None,
    #     tokenizer=tokenizer,
    #     data_collator=data_collator,
    #     compute_metrics=compute_metrics
    # )

    trainer.train()
    output_dir = model_args.output_dir + '_' + str(custom_args.samples) + '/' + str(custom_args.bank_ids) 

    model.save_pretrained(output_dir)
    # results = {}
    # if model_args.do_eval:
    #     # print(eval_dataset)
    #     # print(eval_datasets)
    #     batch_size = 8 
    #     for i in range(0, len(eval_datasets), batch_size):
    #         batch = eval_datasets[i: i + batch_size]
    #         inputs = tokenizer(batch['source'], return_tensors='pt',padding=padding, truncation=True, max_length=max_source_length) 
    #         inputs = inputs.to(model.device)
    #         model_dtype = next(model.parameters()).dtype
    #         print(f"Model's data type: {model_dtype}")
    #         tensor_dtype = inputs.input_ids.dtype
    #         print(f"Tensor's data type: {tensor_dtype}")
    #         # print(inputs)
    #         # print(model)
    #         outputs = model.generate(**inputs,
    #             max_length=512, do_sample=True)
    #         print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    #         # print(batch['input_ids'])
    #         # print(batch['instructions'])
    #     # print(eval_dataset)
    #     # print(train_dataset)
    #     # exit()
    #     # predictions = trainer.predict(eval_dataset).predictions
    #     # # print(predictions)
    #     # # print(predictions[1].shape)
    #     # # for pred in predictions[1]:
    #     # #     print(pred.shape)
    #     # predicted_token_ids = np.argmax(predictions[0], axis=-1)
    #     # predictions = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions[0]]
    #     print(predicted_token_ids)
    #     decoded_preds = tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)
    #     print(decoded_preds)
    #     labels = trainer.predict(eval_dataset).label_ids
    #     print(np.array(labels).shape)
        
    #     # Replace -100 in the labels as we can't decode them.
    #     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    #     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    #     print(decoded_labels)
    #     exit()
    # #     logger.info("*** Evaluate ***")
    # #     print(len(eval_datasets))
    #     # for task, eval_dataset in eval_datasets.items():
    #     #     print(task)
    #     #     print(eval_dataset)
    # #     #     print(eval_)
    # #     metrics = trainer.evaluate(eval_dataset=eval_dataset)
    # #     trainer.log_metrics("eval", metrics)
    # #     results['evaluate'] = metrics

    # # # Test
    # # if model_args.do_predict:
    # #     logger.info("*** Test ***")
    # #     #for task, test_dataset in test_datasets.items():
    # #     metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    # #     trainer.log_metrics("test", metrics)
    # #     results['test'] = metrics
    # #os.makedirs("delta_model")


if __name__ == "__main__":
    train()
