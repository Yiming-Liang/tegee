import os
from dataclasses import field, dataclass
from typing import Optional, Any
from transformers import TrainingArguments

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
    def __init__(self, val_tasknames, test_tasknames):
        self.val_tasknames = val_tasknames 
        self.test_tasknames = test_tasknames 
        self.addins = True

    def get_test(self):
        print(self.test_tasknames)
        test_dts = []  
        for taskname in (self.test_tasknames):
            with open(taskname, encoding='UTF-8') as f:
                task = json.load(f)
            ins = task['Instances']
            length = len(ins)
            df = pd.DataFrame(ins)
            if len(task['Definition']) > 1:
                merged = " ".join([string for string in task['Definition'] if string])
                task['Definition'] = [merged]
            df['instructions'] = task['Definition']*length
            if self.addins:     #add instructions
                df['source'] = df['instructions'].map(list_to_string)  + df['input']
                df['target'] = df['output'].map(list_to_string)
            else:
                df['source'] = df['input']
                df['target'] = df['output'].map(list_to_string)
            _, test_dt = train_test_split(df, random_state=0, train_size=0.9)
            test_dts.append(test_dt)
            
            # train_dt , val_dt = train_test_split(trainval_dt, random_state=42, train_size=0.9)
            # if len(train_dt)>=args.samples:
            #     train_dt = train_dt[:args.samples]
        print(len(test_dts))
        concated_dt = pd.concat(test_dts,ignore_index=True) 
        print(concated_dt.shape)
        print(concated_dt.iloc[0]["source"])
        print(concated_dt.iloc[0]["target"])
        print(concated_dt.iloc[1]["source"])
        return concated_dt 

    def get_val(self):
        print(self.val_tasknames)
        # print(self.taskname)
        test_dts = []  
        for taskname in (self.val_tasknames):
            with open(taskname, encoding='UTF-8') as f:
                task = json.load(f)
            ins = task['Instances']
            length = len(ins)
            df = pd.DataFrame(ins)
            if len(task['Definition']) > 1:
                merged = " ".join([string for string in task['Definition'] if string])
                task['Definition'] = [merged]
            df['instructions'] = task['Definition']*length
            if self.addins:     #add instructions
                df['source'] = df['instructions'].map(list_to_string)  + df['input']
                df['target'] = df['output'].map(list_to_string)
            else:
                df['source'] = df['input']
                df['target'] = df['output'].map(list_to_string)
            _, test_dt = train_test_split(df, random_state=42, train_size=0.9)
            test_dts.append(test_dt)
            print(test_dt.shape)
            # train_dt , val_dt = train_test_split(trainval_dt, random_state=42, train_size=0.9)
            # if len(train_dt)>=args.samples:
            #     train_dt = train_dt[:args.samples]
        test_dts = [test_dt[:args.samples] for test_dt in test_dts if len(test_dt) >= args.samples]
        print(len(test_dts))
        concated_dt = pd.concat(test_dts,ignore_index=True) 
        print(concated_dt)
        return concated_dt 


def train():
    parser = RemainArgHfArgumentParser((TrainingArguments))
    model_args = parser.parse_args_into_dataclasses()[0]
    print(model_args)
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


    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        load_in_8bit=model_args.load_in_8bit,
        use_cache=False,
        torch_dtype=torch.float16,
        cache_dir=model_args.cache_dir,
        device_map=device_map,
    )

    if model_args.load_in_8bit:
        # print('here')
        model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules=model_args.lora_target_modules,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    # data_fp = '{}/task{}.json'.format(bank_ids) 
    # print(data_fp)
    dt = get_data(model_args, model_args.data_paths[0])    #get data
    test_dataset = dt.get_test()
    val_dataset = dt.get_val()
    test_datasets = Dataset.from_pandas(test_dataset)
    print('****' * 100)
    eval_datasets = Dataset.from_pandas(val_dataset)
    column_names = ['source', 'target']


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

    model.save_pretrained(model_args.output_dir)
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
