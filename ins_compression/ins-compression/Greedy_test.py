import os
from dataclasses import field, dataclass
from typing import Optional, Any
from transformers import TrainingArguments
import string

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
import transformers
from peft import PeftModel
parser = argparse.ArgumentParser()
parser.add_argument("--data_fp", default='', type=str,
                    help="the datapath of meta files")
parser.add_argument("--info_dict", default='', type=str,
                    help="the datapath of meta files")
parser.add_argument("--output_dir", default='', type=str,
                    help="the datapath of meta files")
parser.add_argument("--padding", default='max_length', type=str,
                    help="the datapath of meta files")
parser.add_argument("--max_source_length", default=512, type=int,
                    help="length of input sentence")
parser.add_argument("--model_max_length", default=512, type=int,
                    help="length of input sentence")
parser.add_argument("--samples", default=10, type=int,
                    help="length of input sentence")
parser.add_argument("--bank_ids", default=5, type=int,
                    help="length of input sentence")
parser.add_argument("--max_target_length", default=512, type=int,
                    help="length of output sentence")
parser.add_argument("--lora_save_dir", default="/huangwhai52/gezhang/models/n_logs/greedysoup/t5-base/5/delta_save/0", type=str)
parser.add_argument("--model_path", default="", type=str)
parser.add_argument("--do_eval", default=True, type=bool,
                    help="do val during training")
parser.add_argument("--do_predict", default=True, type=bool,
                    help="test after training")
parser.add_argument("--do_train", default=True, type=bool,
                    help="do train")
args = parser.parse_args()

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

class get_data():
    '''
    load data model
    '''
    def __init__(self,args,taskname, samples):
        self.taskname = taskname 
        self.do_train = args.do_train
        self.do_val = args.do_eval
        self.do_test = args.do_predict
        self.samples = samples

        self.addins = True

 

    def make(self):
        print(self.taskname)
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
            df['source'] = df['instructions'].map(list_to_string)  + df['input']
            df['target'] = df['output'].map(list_to_string)
        else:
            df['source'] = df['input']
            df['target'] = df['output'].map(list_to_string)
        trainval_dt, test_dt = train_test_split(df, random_state=42, train_size=0.9)
        train_dt , val_dt = train_test_split(trainval_dt, random_state=42, train_size=0.9)
        if len(train_dt)>=self.samples:
            train_dt = train_dt[:self.samples]
        return train_dt, test_dt, val_dt

def eval():
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path,
        cache_dir='./',
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False
    )
    with open(args.info_dict) as fp:
        bank = json.load(fp)
    data_fp = f"{args.data_fp}/{bank[str(args.bank_ids)][0]}"
    dt = get_data(args, data_fp, args.samples)    #get data
    train_dataset, test_dataset, val_dataset = dt.make()
    train_datasets = Dataset.from_pandas(train_dataset)
    test_datasets = Dataset.from_pandas(test_dataset)
    eval_datasets = Dataset.from_pandas(val_dataset)
    base_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(args.model_path, device_map='auto' )
    lora_dir = args.lora_save_dir + '_' + str(args.samples) + '/' + str(args.bank_ids) + '/' 
    peft_model = PeftModel.from_pretrained(base_model, lora_dir)
    print(eval_datasets)

    device_map = "auto"
    def evaluations(eval_datasets, peft_model, tokenizer):
        batch_size = 40 
        predictions = []
        decoded_labels = []
        for i in range(0, len(eval_datasets), batch_size):
            batch = eval_datasets[i: i + batch_size]
            inputs = tokenizer(batch['source'], return_tensors='pt',padding=args.padding, truncation=True, max_length=args.max_source_length) 
            inputs = inputs.to(peft_model.device)
            outputs = peft_model.generate(**inputs,
                max_length=512, do_sample=True)
            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions += decoded_preds
            print(decoded_preds)
            # batch['']
            print(batch['target'])
            decoded_labels += batch['target']
        from rouge_score import rouge_scorer
        def normalize_answer(s):
            """Lower text and remove punctuation, and extra whitespace."""

            def white_space_fix(text):
                return ' '.join(text.split())

            def remove_punc(text):
                exclude = set(string.punctuation)
                return ''.join(ch for ch in text if ch not in exclude)

            def lower(text):
                return text.lower()

            return white_space_fix(remove_punc(lower(s)))
        def exact_match_judge(prediction, ground_truth):
            return (normalize_answer(prediction) == normalize_answer(ground_truth))
        
        def roughl(prediction, ground_truth):
            scorer = rouge_scorer.RougeScorer(['rougeL','rouge1','rouge2'], use_stemmer=True)
            scores = scorer.score(prediction=prediction, target=ground_truth)
            return scores["rougeL"].fmeasure, scores["rougeL"].precision, scores['rougeL'].recall, scores["rouge1"].fmeasure, scores["rouge1"].precision, scores['rouge1'].recall,scores["rouge2"].fmeasure, scores["rouge2"].precision, scores['rouge2'].recall
        exact_match_score = 0
        rouge_l_score = 0
        rouge_l_fmeasure = 0
        rouge_l_precision = 0
        rouge_l_recall = 0
        rouge_1_fmeasure = 0
        rouge_1_precision = 0
        rouge_1_recall = 0
        rouge_2_fmeasure = 0
        rouge_2_precision = 0
        rouge_2_recall = 0
        pred_label = []
        for pred, leb in zip(predictions, decoded_labels):
            if exact_match_judge(pred,leb):
                exact_match_score+=1
            rouge_l_fmeasure = rouge_l_fmeasure + roughl(pred, leb)[0]
            rouge_l_precision = rouge_l_precision + roughl(pred, leb)[1]
            rouge_l_recall = rouge_l_recall + roughl(pred, leb)[2]
            rouge_1_fmeasure = rouge_1_fmeasure + roughl(pred, leb)[3]
            rouge_1_precision = rouge_1_precision + roughl(pred, leb)[4]
            rouge_1_recall = rouge_1_recall + roughl(pred, leb)[5]
            rouge_2_fmeasure = rouge_2_fmeasure + roughl(pred, leb)[6]
            rouge_2_precision = rouge_2_precision + roughl(pred, leb)[7]
            rouge_2_recall = rouge_2_recall + roughl(pred, leb)[8]
            if not pred in pred_label:
                pred_label.append(pred)
        exact_match_score = 100.0 * exact_match_score/len(decoded_labels)
        rouge_l_fmeasure = 100.0 * rouge_l_fmeasure/ len(decoded_labels)
        rouge_l_precision = 100.0*rouge_l_precision/len(decoded_labels)
        rouge_l_recall = 100.0 * rouge_l_recall/len(decoded_labels)
        rouge_1_fmeasure = 100.0 * rouge_1_fmeasure/ len(decoded_labels)
        rouge_1_precision = 100.0*rouge_1_precision/len(decoded_labels)
        rouge_1_recall = 100.0 * rouge_1_recall/len(decoded_labels)
        rouge_2_fmeasure = 100.0 * rouge_2_fmeasure/ len(decoded_labels)
        rouge_2_precision = 100.0*rouge_2_precision/len(decoded_labels)
        rouge_2_recall = 100.0 * rouge_2_recall/len(decoded_labels)
        print("fmeasure", rouge_l_fmeasure, "pre", rouge_l_precision, "recall", rouge_l_recall)
        print("fmeasure", rouge_1_fmeasure, "pre", rouge_1_precision, "recall", rouge_1_recall)
        print("fmeasure", rouge_2_fmeasure, "pre", rouge_2_precision, "recall", rouge_2_recall)
        result = {}
        result.update({"exact_match": exact_match_score})
        result.update( {"rouge_L_fmeasure": rouge_l_fmeasure})
        result.update( {"rouge_L_precision": rouge_l_precision})
        result.update( {"rouge_L_recall": rouge_l_recall})
        result.update( {"rouge_1_fmeasure": rouge_1_fmeasure})
        result.update( {"rouge_1_precision": rouge_1_precision})
        result.update( {"rouge_1_recall": rouge_1_recall})
        result.update( {"rouge_2_fmeasure": rouge_2_fmeasure})
        result.update( {"rouge_2_precision": rouge_2_precision})
        result.update( {"rouge_2_recall": rouge_2_recall})
        result.update({"eval_average_metrics":exact_match_score})
        # result.update({"types of lebs":len(pred_label)})
        # result.update({"model_soup":str(args.model_list)})
        result.update({"data_path":args.data_fp})
        print(result)
        return result
    results = {}
    if args.do_eval:
        results['valiation'] = evaluations(eval_datasets, peft_model, tokenizer)
    if args.do_predict:
        results['prediction'] = evaluations(test_datasets, peft_model, tokenizer)
    
    result_json = args.output_dir + '_' + str(args.samples) + '/' + str(args.bank_ids) + '/' + 'results.json'
    with open(result_json, 'w') as f:
        json.dump(results, f)
        # return result

if __name__ == "__main__":
    eval()