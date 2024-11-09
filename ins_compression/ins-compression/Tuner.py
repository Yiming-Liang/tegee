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
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

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

# parser = transformers.HfArgumentParser(TrainingArguments)
# args = parser.parse_args_into_dataclasses()[0]
# parser_command = argparse.ArgumentParser()


# def compute_metrics(eval_pred):
#     # parser = RemainArgHfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
#     # model_args, data_args, training_args, delta_args = parser.parse_json_file(json_file=args.config)
#     tokenizer = transformers.AutoTokenizer.from_pretrained(
#         # model_args.model_name_or_path,
#         '/cpfs/29cd2992fe666f2a/shared/public/self-ins/t5-small',
#         cache_dir='./cache',
#         model_max_length=512,
#         padding_side="right",
#         use_fast=False
#     )
#     # tokenizer = AutoTokenizer.from_pretrained(
#     #     model_args.model_name_or_path,
#     #     cache_dir=model_args.cache_dir,
#     #     use_fast=model_args.use_fast_tokenizer,
#     #     revision=model_args.model_revision,
#     #     use_auth_token=True if model_args.use_auth_token else None,
#     # )
#     print(eval_pred)
    
#     predictions = eval_pred.predictions
#     labels = eval_pred.label_ids
#     print('-----')
#     decoded_preds = tokenizer.batch_decode(predictions[1], skip_special_tokens=True)
#     print(decoded_preds)
    
#     # Replace -100 in the labels as we can't decode them.
#     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
#     from rouge_score import rouge_scorer
#     def normalize_answer(s):
#         """Lower text and remove punctuation, and extra whitespace."""

#         def white_space_fix(text):
#             return ' '.join(text.split())

#         def remove_punc(text):
#             exclude = set(string.punctuation)
#             return ''.join(ch for ch in text if ch not in exclude)

#         def lower(text):
#             return text.lower()

#         return white_space_fix(remove_punc(lower(s)))
#     def exact_match_judge(prediction, ground_truth):
#         return (normalize_answer(prediction) == normalize_answer(ground_truth))
    
#     def roughl(prediction, ground_truth):
#         scorer = rouge_scorer.RougeScorer(['rougeL','rouge1','rouge2'], use_stemmer=True)
#         scores = scorer.score(prediction=prediction, target=ground_truth)
#         return scores["rougeL"].fmeasure, scores["rougeL"].precision, scores['rougeL'].recall, scores["rouge1"].fmeasure, scores["rouge1"].precision, scores['rouge1'].recall,scores["rouge2"].fmeasure, scores["rouge2"].precision, scores['rouge2'].recall
#     exact_match_score = 0
#     rouge_l_score = 0
#     rouge_l_fmeasure = 0
#     rouge_l_precision = 0
#     rouge_l_recall = 0
#     rouge_1_fmeasure = 0
#     rouge_1_precision = 0
#     rouge_1_recall = 0
#     rouge_2_fmeasure = 0
#     rouge_2_precision = 0
#     rouge_2_recall = 0
#     pred_label = []
#     for pred, leb in zip(decoded_preds, decoded_labels):
#         if exact_match_judge(pred,leb):
#             exact_match_score+=1
#         rouge_l_fmeasure = rouge_l_fmeasure + roughl(pred, leb)[0]
#         rouge_l_precision = rouge_l_precision + roughl(pred, leb)[1]
#         rouge_l_recall = rouge_l_recall + roughl(pred, leb)[2]
#         rouge_1_fmeasure = rouge_1_fmeasure + roughl(pred, leb)[3]
#         rouge_1_precision = rouge_1_precision + roughl(pred, leb)[4]
#         rouge_1_recall = rouge_1_recall + roughl(pred, leb)[5]
#         rouge_2_fmeasure = rouge_2_fmeasure + roughl(pred, leb)[6]
#         rouge_2_precision = rouge_2_precision + roughl(pred, leb)[7]
#         rouge_2_recall = rouge_2_recall + roughl(pred, leb)[8]
#         if not pred in pred_label:
#             pred_label.append(pred)
#     exact_match_score = 100.0 * exact_match_score/len(decoded_labels)
#     rouge_l_fmeasure = 100.0 * rouge_l_fmeasure/ len(decoded_labels)
#     rouge_l_precision = 100.0*rouge_l_precision/len(decoded_labels)
#     rouge_l_recall = 100.0 * rouge_l_recall/len(decoded_labels)
#     rouge_1_fmeasure = 100.0 * rouge_1_fmeasure/ len(decoded_labels)
#     rouge_1_precision = 100.0*rouge_1_precision/len(decoded_labels)
#     rouge_1_recall = 100.0 * rouge_1_recall/len(decoded_labels)
#     rouge_2_fmeasure = 100.0 * rouge_2_fmeasure/ len(decoded_labels)
#     rouge_2_precision = 100.0*rouge_2_precision/len(decoded_labels)
#     rouge_2_recall = 100.0 * rouge_2_recall/len(decoded_labels)
#     print("fmeasure", rouge_l_fmeasure, "pre", rouge_l_precision, "recall", rouge_l_recall)
#     print("fmeasure", rouge_1_fmeasure, "pre", rouge_1_precision, "recall", rouge_1_recall)
#     print("fmeasure", rouge_2_fmeasure, "pre", rouge_2_precision, "recall", rouge_2_recall)
#     result = {}
#     result.update({"exact_match": exact_match_score})
#     result.update( {"rouge_L_fmeasure": rouge_l_fmeasure})
#     result.update( {"rouge_L_precision": rouge_l_precision})
#     result.update( {"rouge_L_recall": rouge_l_recall})
#     result.update( {"rouge_1_fmeasure": rouge_1_fmeasure})
#     result.update( {"rouge_1_precision": rouge_1_precision})
#     result.update( {"rouge_1_recall": rouge_1_recall})
#     result.update( {"rouge_2_fmeasure": rouge_2_fmeasure})
#     result.update( {"rouge_2_precision": rouge_2_precision})
#     result.update( {"rouge_2_recall": rouge_2_recall})
#     result.update({"eval_average_metrics":exact_match_score})
#     result.update({"types of lebs":len(pred_label)})
#     result.update({"model_soup":str(args.model_list)})
#     result.update({"data_path":args.data_fp})
#     return result


class get_data():
    '''
    load data model
    '''
    def __init__(self,args, data_fp):
        self.taskname = data_fp
        self.do_train = args.do_train
        self.do_val = args.do_eval
        self.do_test = args.do_predict

        self.addins = True

 

    def make(self):
        print(self.taskname)
        with open(self.taskname, encoding='UTF-8') as f:
            task = json.load(f)
        ins = task['Instances']
        length = len(ins)
        print(length)
        df = pd.DataFrame(ins)
        print(df.shape)
        print(task['Definition'])
        print(len(task['Definition']))
        # print(task['Definition'] * length)
        # print(df['instructions'])
        
        if len(task['Definition']) > 1:
            merged = " ".join([string for string in task['Definition'] if string])
            task['Definition'] = [merged]
        dd = task['Definition']*length
        print(len(dd))
        df['instructions'] = dd
        if self.addins:     #add instructions
            df['source'] = df['instructions'].map(list_to_string)  + df['input']
            df['target'] = df['output'].map(list_to_string)
        else:
            df['source'] = df['input']
            df['target'] = df['output'].map(list_to_string)
        trainval_dt, test_dt = train_test_split(df, random_state=42, train_size=0.9)
        train_dt , val_dt = train_test_split(trainval_dt, random_state=42, train_size=0.9)
        return train_dt, test_dt, val_dt



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

    print_trainable_parameters(model)
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
    print_trainable_parameters(model)


    # def count_trainable_parameters(model):
    #     return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # def count_trainable_parameters(model):
    #     return sum(p.numel() for p in model.parameters())

    # Counting the trainable parameters
    # num_trainable_params = count_trainable_parameters(model)
    # num_params = count_trainable_parameters(model)
    # print(' trainable ', num_trainable_params)
    # print(' all ', num_params)
    # exit()
    # data_fp = '{}/task{}.json'.format(bank_ids) 
    # print(data_fp)
    dt = get_data(model_args, model_args.data_paths[0])    #get data
    train_dataset, test_dataset, val_dataset = dt.make()
    # print(len(train_dataset))
    # for d in train_dataset['input']:
    #     print(d)
    # # print(train_dataset[0])
    # exit()
    train_datasets = Dataset.from_pandas(train_dataset)
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
