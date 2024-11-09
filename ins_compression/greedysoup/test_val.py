# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from opendelta import AutoDeltaConfig, AutoDeltaModel, LowRankAdapterModel
from transformers.trainer_utils import is_main_process, get_last_checkpoint
import sys
import argparse
import json
import string
import numpy as np
import torch
import transformers
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments
from datasets import Dataset, load_metric
import functools
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
import numpy as np
from examples_seq2seq.seq2seq_trainer import Seq2SeqTrainer #预装在了opendelta的package里，sourcecode被重命名为train_files
from examples_seq2seq.trainers.model_args import ModelArguments
from examples_seq2seq.trainers.trainer_args import TrainingArguments, DataTrainingArguments
import os
import logging
import collections
set_seed(42)
logger = logging.getLogger(__name__)
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

def compute_metrics(eval_pred):
    parser = RemainArgHfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args, delta_args = parser.parse_json_file(json_file=args.config)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    print(eval_pred)
    
    predictions = eval_pred[0]
    labels = eval_pred[1]
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    print(decoded_preds)
    
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
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
    for pred, leb in zip(decoded_preds, decoded_labels):
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
    result.update({"types of lebs":len(pred_label)})
    result.update({"model_soup":str(args.model_list)})
    result.update({"data_path":args.data_fp})
    return result


class get_data():
    '''
    load data model
    '''
    def __init__():
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

def main(args):
    parser = RemainArgHfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args, delta_args = parser.parse_json_file(json_file=args.config)
    training_args.output_dir = args.output_dir 
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    config.dropout_rate = 0.0
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
    )

    # Detecting last checkpoint.
    # last_checkpoint = None
    # if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
    #     last_checkpoint = get_last_checkpoint(training_args.output_dir)
    #     print("#### last_checkpoint ", last_checkpoint)
    #     if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
    #         '''
    #         raise ValueError(
    #             f"Output directory ({training_args.output_dir}) already exists and is not empty. "
    #             "Use --overwrite_output_dir to overcome."
    #         )
    #         '''
    #         pass
    #     elif last_checkpoint is not None:
    #         logger.info(
    #             f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
    #             "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
    #         )
    set_seed(training_args.seed)
    #model average
    model_paths = []
    with open(args.bank_config) as fp:
        bank = json.load(fp)
    data_fp = f"{args.data_fp}/{bank[str(args.bank_ids)][0]}"
    test_fps = [data_fp] 
    val_fps = []
    for val_fp in bank[str(args.bank_ids)][1]: 
        val_fps.append( f"{args.data_fp}/{val_fp}")
    model_fp = f"{args.model_lora_fp}/{args.bank_ids}"
    print(model_fp)
    # args.model_list = bank[str(args.bank_ids)][1]
    # for md in args.model_list: #initialize a model
    #     model_paths.append(f"{args.model_path}/{md[:-5]}")
    # print(model_paths, data_fp)
    # exit()
    # NUM_MODELS = len(model_paths)
    # for j, model_path in enumerate(model_paths):
    #             state_dict = AutoModelForSeq2SeqLM.from_pretrained(
    #     model_args.model_name_or_path,
    #     from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #     config=config,
    #     cache_dir=model_args.cache_dir,
    #     revision=model_args.model_revision,
    #     use_auth_token=None,
    #     ignore_mismatched_sizes=True
    # )
    #             state_dict.resize_token_embeddings(len(tokenizer))
    #             print(state_dict.state_dict())
    #             print('-------' * 20)
    #             delta = AutoDeltaModel.from_finetuned(model_path, backbone_model=state_dict)
    #             state_dict = state_dict.state_dict()
    #             print(state_dict)
    #             exit()

    #             if j == 0:
    #                 uniform_soup = {k : v * (1./NUM_MODELS) for k, v in state_dict.items()}
    #             else:
    #                 uniform_soup = {k : v * (1./NUM_MODELS) + uniform_soup[k] for k, v in state_dict.items()}
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=True
    )
    model.resize_token_embeddings(len(tokenizer))
    delta_model = AutoDeltaModel.from_finetuned(model_fp, backbone_model=model)
    model.eval()
    print(delta_model.state_dict())
    print("-------------------")
    # # print('uniform_soup', uniform_soup)
    # model.load_state_dict(model.state_dict())
    # delta_model.freeze_module(set_state_dict=True)
    # exit()
    #print(model.state_dict()["decoder.block.8.layer.0.SelfAttention.q.lora.lora_A"])
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    # if is_main_process(training_args.local_rank):
    #     transformers.utils.logging.set_verbosity_info()
    # logger.info("Training/evaluation parameters %s", training_args)
                           
    # # model parallelize
    # # if hasattr(training_args, "model_parallel") and training_args.model_parallel:
    # #     logger.info('parallelize model!')
    # model.parallelize()
    dt = get_data(args, val_fps, test_fps)    #get data
    test_dataset = dt.get_test()
    val_dataset = dt.get_val()
    # exit()
    # print(len(train_dataset))
    # for d in train_dataset['input']:
    #     print(d)
    # # print(train_dataset[0])
    # exit()
    # train_datasets = Dataset.from_pandas(train_dataset)
    # print(train_datasets[0]['target'])
    test_datasets = Dataset.from_pandas(test_dataset)
    # print(test_datasets)
    # print(test_datasets[0]['source'])
    # print(test_datasets[0]['target'])
    print('****' * 100)
    eval_datasets = Dataset.from_pandas(val_dataset)
    # print(eval_datasets[0]['source'])
    # print(eval_datasets[0]['target'])
    column_names = ['source', 'target']
    # exit()


    def preprocess_function(examples):
        '''
        inheritate from opendelta example
        '''
        model_inputs = tokenizer([s for s in examples['source']], max_length=args.max_source_length,
                                 padding=args.padding, truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer([t for t in examples['target']], max_length=args.max_target_length, padding=args.padding, truncation=True)
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if args.padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # train_dataset = train_datasets.map(
    #             functools.partial(preprocess_function),
    #             batched=True,
    #             remove_columns=column_names, 
    #         )

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
    
    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        delta_args=delta_args,
        train_dataset=None if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # trainer.add_callback(MyCallback(trainer_args=training_args, delta_args=delta_args, model_args=model_args))


    
    performance_metrics = {}
    # Training
    results = {}
    if args.do_val:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=test_dataset,
               max_length=data_args.val_max_target_length, num_beams=data_args.num_beams,
            )
        trainer.log_metrics("eval", metrics)
        results['evaluate'] = metrics
    
    
    # Test
    if args.do_test:
        logger.info("*** Test ***")
        # print(test_dataset['input_ids'])
        # print(test_dataset['labels'])
        metrics = trainer.evaluate(eval_dataset=eval_dataset,
              max_length=data_args.test_max_target_length, num_beams=data_args.num_beams,
            #   metric_key_prefix="test"
            )
        trainer.log_metrics("test", metrics)
        results['test'] = metrics
    #os.makedirs("delta_save")
    # print(delta_model.state_dict())
    # if delta_args.delta_type.lower() != "none":
    #     delta_model.save_finetuned('{}/{}/'.format(args.lora_save_dir, args.bank_ids))
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_fp", default='', type=str,
                        help="the datapath of meta files")
    parser.add_argument("--padding", default='max_length', type=str,
                        help="the datapath of meta files")
    parser.add_argument("--do_train", default=False, type=bool,
                        help="do train")
    parser.add_argument("--max_source_length", default=512, type=int,
                        help="length of input sentence")
    parser.add_argument("--max_target_length", default=512, type=int,
                        help="length of output sentence")
    parser.add_argument("--do_val", default=True, type=bool,
                        help="do val during training")
    parser.add_argument("--do_test", default=True, type=bool,
                        help="test after training")
    parser.add_argument("--ignore_pad_token_for_loss", default=True, type=bool,
                        help="")
    parser.add_argument("--config", default='/huangwhai52/gezhang/models/5.5/trainlarge/lr_0.0003/lorabase.json', type=str)
    parser.add_argument("--output_dir", default="outputs/lora/t5-base/", type=str)
    parser.add_argument("--lora_save_dir", default="/huangwhai52/gezhang/models/n_logs/greedysoup/t5-base/5/delta_save/0", type=str)
    parser.add_argument("--loraconfig", default={}, type=dict)
    parser.add_argument("--model_path", default="", type=str)
    parser.add_argument("--model_lora_fp", default="", type=str)
    parser.add_argument("--model_list", nargs = '+')
    parser.add_argument("--bank_config", default = "/huangwhai52/gezhang/models/5.5/greedysoup_large/5/info_dict.json", type=str)
    parser.add_argument("--bank_ids", default = 0, type=int)
    parser.add_argument("--samples", default = 5, type=int)
    args = parser.parse_args()
    result = main(args)
    import json
    with open(args.lora_save_dir +'/{}/'.format(args.bank_ids) +"collect_result_rerun.jsonl", 'a') as fout:
        string = json.dumps(result, indent=4,sort_keys=True)
        fout.write(string+"\n")
    print(result)
