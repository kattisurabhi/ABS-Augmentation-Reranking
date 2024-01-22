import sys
sys.path.insert(0, '/home/data')
import pickle
from copy import copy
from transformers import T5Tokenizer, T5ForConditionalGeneration, default_data_collator, PegasusForConditionalGeneration, AutoTokenizer
from transformers import BartForConditionalGeneration
from tqdm import tqdm
import nltk
import os
import argparse
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import pandas as pd

import spacy
nlp = spacy.load("en_core_web_sm")
from copy import copy
import math
nltk.download('punkt')

def main():
    def mask_input_np(org_document, ratio=0.5, is_bart=True):
        masked_input, nnc, noun_phrases = generate_sent_input_all(org_document, is_bart=False)
        model_input, target = fill_phrases_by_sent(masked_input, noun_phrases, nnc, ratio=ratio, is_bart=is_bart)
        return model_input

    def mask_input_split(inputs, ratio=0.3, is_bart=True):
        inputs_ = inputs.split(' ')
        picked_idxs = np.random.choice(list(range(len(inputs_))), int(len(inputs_)*ratio), replace=False)
        for i, idx in enumerate(picked_idxs):
            if is_bart:
                inputs_[idx] = '<mask>'
            else:
                inputs_[idx] = '<extra_id_'+str(i)+'>'
        return ' '.join(inputs_)

    def generate_sent_input_all(sent, nmask=0, is_bart=False):
        masked_phrases = []
        try:
            doc = nlp(sent)
            noun_chunks = []
            noun_chunk_pos = []
            for chunk in doc.noun_chunks:
                noun_chunks.append(chunk.text)
                noun_chunk_pos.append((chunk.start, chunk.end))

            all_noun_chunks = copy(noun_chunks)

            if len(noun_chunks) > 0:
                masked_sent = []
                splitted_sent = [word.text for word in doc]

                for i in range(len(noun_chunk_pos)):
                    cur_start = noun_chunk_pos[i][0]
                    cur_end = noun_chunk_pos[i][1]

                    if i == 0:#cur_start != 0:
                        masked_sent += splitted_sent[:cur_start]

                    if i == len(noun_chunk_pos)-1:
                        next_start = len(splitted_sent)
                    else:
                        next_start = noun_chunk_pos[i+1][0]

                    if is_bart:
                        masked_sent += ['<mask>']
                    else:
                        masked_sent += ['<extra_id_'+str(nmask)+'>']
                    
                    masked_phrases += [' '.join(splitted_sent[cur_start:cur_end])]
                    nmask += 1

                    masked_sent += splitted_sent[cur_end:next_start]
                    masked_sent_ = ' '.join(masked_sent)
                
                return masked_sent_, nmask, all_noun_chunks
            
            
            else:
                return sent, 0, []
                
        except:
            return sent, 0, []
        
    def fill_phrases_by_sent(masked_sent, noun_chunks, nnc, ratio=0.9, is_bart=False, max_n=100):
        masked_sent_ = copy(masked_sent)

        all_pidxs = []
        all_nidxs = []


        for sent in nltk.sent_tokenize(masked_sent_):        
            masked_idxs = [int(x.split('>')[0].split('_')[-1]) for x in sent.split('<') if x.startswith('extra_')]
            nnc = len(masked_idxs)
            nfill = max(0, math.ceil(nnc*(1-ratio)))
            if ratio == 1.0:
                nfill = 0
            pidxs = sorted(np.random.choice(masked_idxs, nfill, replace=False))
            nidxs = [x for x in masked_idxs if x not in pidxs]
            
            all_pidxs += pidxs
            all_nidxs += nidxs

        for idx in all_pidxs:
            masked_sent_ = masked_sent_.replace('<extra_id_'+str(idx)+'>', noun_chunks[idx])
            
        noun_chunks_picked = [noun_chunks[idx] for idx in all_nidxs]
        target = ''


        for i, idx in enumerate(all_nidxs):
            if is_bart:
                masked_sent_ = masked_sent_.replace('<extra_id_'+str(idx)+'>', '<mask>')
            else:           
                masked_sent_ = masked_sent_.replace('<extra_id_'+str(idx)+'>', '<extra_id_'+str(i)+'>')
            
            each_target = '<extra_id_'+str(i)+'> '+noun_chunks[idx]+' ' 
            target += each_target
            
        if (is_bart == False) and len(all_nidxs) == max_n:
            cut_idx = masked_sent_.index('<extra_id_99>')+len('<extra_id_99>')
            masked_sent_ = masked_sent_[:cut_idx]
            
        return masked_sent_, target

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", default="ccdv/cnn_dailymail", type=str)
    parser.add_argument("--dataset_config_name", default="3.0.0", type=str)

    parser.add_argument("--input_file", default="None", type=str)
    parser.add_argument("--output_file", default="test.pkl", type=str)
    parser.add_argument("--ckpt_dir", default='/content/drive/MyDrive/train_generate_summary/output', type=str)

    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--tgt_max_length", default=128, type=int)
    parser.add_argument("--src_max_length", default=1024, type=int)
    parser.add_argument("--num_beams", default=8, type=int)

    parser.add_argument("--num_cpus", default=2, type=int)
    parser.add_argument("--mask_ratio1", default=0.0, type=float)
    parser.add_argument("--mask_ratio2", default=0.0, type=float)
    parser.add_argument("--mask_type", default='np', type=str)
    parser.add_argument("--odd_even", default='odd', type=str)
    parser.add_argument("--num_return_sequences", default=1, type=int)
    parser.add_argument("--diversity_penalty", default=0.1, type=float)
    parser.add_argument("--num_beam_groups", default=8, type=int)
    args = parser.parse_args()

    raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)

    num_beams = args.num_beams
    model_path = args.ckpt_dir
    num_return_sequences = args.num_return_sequences
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = BartForConditionalGeneration.from_pretrained(model_path)

    model.eval()

    def preprocess_function(examples):
        mask_ratio1 = args.mask_ratio1
        mask_ratio2 = args.mask_ratio2
        if args.input_file != 'None':
            masked_input =  examples['masked_articles']
            noun_phrases = examples['nchunks_articles']
            nnc = examples['nnc']
            summaries = examples['summary']
            

            inputs_2 = [fill_phrases_by_sent(x, y, z, ratio=args.mask_ratio2, is_bart=bool(args.is_bart))[0] for x,y,z in zip(masked_input, noun_phrases, nnc)]
            inputs_1 = [mask_input_np(x, ratio=args.mask_ratio1) for x in summaries]
        
            inputs = ['summary: '+x+' article: '+y for x,y in zip(inputs_1, inputs_2)]   
        else:
            inputs_1 = examples['highlights']
            inputs_2 = examples['article']
                        
            if args.mask_ratio1 != 0.0:
                if args.mask_type == 'np':
                    inputs_1 = [mask_input_np(x, ratio=args.mask_ratio1) for x in inputs_1]
                    inputs_2 = [mask_input_np(x, ratio=args.mask_ratio2) for x in inputs_2]

                else:
                    inputs_1 = [mask_input_split(x, ratio=args.mask_ratio1) for x in inputs_1]
                    inputs_2 = [mask_input_split(x, ratio=args.mask_ratio2) for x in inputs_2]
            inputs = ['summary: '+x+' article: '+y for x,y in zip(inputs_1, inputs_2)]

        model_inputs = tokenizer(inputs, max_length=args.src_max_length, padding='max_length', truncation=True)
        return model_inputs


    filled_summaries = []
    all_inputs = []

    if args.input_file != 'None':
        with open(args.input_file, 'rb') as f:
            infer_data = pickle.load(f)
        infer_data = Dataset.from_dict(infer_data)
        column_names = infer_data.column_names

    else:
        infer_data = raw_datasets['test'] 
        column_names = infer_data.column_names
        
        if args.odd_even != 'all':
            if args.odd_even == 'even':
                subset = [x for i,x in enumerate(infer_data) if i % 2 == 0]
            else:
                subset = [x for i,x in enumerate(infer_data) if i % 2 == 1]


            subset_ = {}
            for key in infer_data[0]:
                subset_[key] = [x[key] for x in subset]
            infer_data = Dataset.from_dict(subset_)

    dataset = infer_data.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on dataset",
        num_proc=args.num_cpus,
        remove_columns=column_names
    )

    dataloader = DataLoader(dataset, collate_fn=default_data_collator, batch_size=args.batch_size, shuffle=False)
    
    def to_list(filled_summaries, num_return_sequences):
        filled_summaries_output=[]
        num_summaries = len(filled_summaries)
        num_lists = num_summaries // num_return_sequences
    
        for i in range(num_lists):
          start_idx = i * num_return_sequences
          end_idx = start_idx + num_return_sequences
          sublist = filled_summaries[start_idx:end_idx]
          filled_summaries_output.append(sublist)
        return filled_summaries_output
    all_outputs = []
    # all_outputs_2 = []
    # all_outputs_3 = []
    # all_outputs_4 = []
    filled_summaries = []
    # filled_summaries_2 = []
    # filled_summaries_3 = []
    # filled_summaries_4 = []
    input_articles = []
    filled_input = []
    for batch in tqdm(dataloader):
        with torch.no_grad():
            outputs = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']
                                        , max_length=args.tgt_max_length, num_beams=num_beams, diversity_penalty=args.diversity_penalty
                                     , num_return_sequences=args.num_return_sequences, num_beam_groups=args.num_beam_groups).cpu()
            
            # outputs_2 = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']
            #                             , max_length=args.tgt_max_length, num_beams=num_beams).cpu()
            
            # outputs_3 = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']
            #                             , max_length=args.tgt_max_length, num_beams=num_beams).cpu()
            
            # outputs_4 = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']
            #                             , max_length=args.tgt_max_length, num_beams=num_beams).cpu()
            
            all_outputs += list(outputs)
            # all_outputs_2 += list(outputs_2)
            # all_outputs_3 += list(outputs_3)
            # all_outputs_4 += list(outputs_4)
            
            filled_summaries += tokenizer.batch_decode(list(outputs), skip_special_tokens=False)
            # filled_summaries_2 += tokenizer.batch_decode(list(outputs_2), skip_special_tokens=False)
            # filled_summaries_3 += tokenizer.batch_decode(list(outputs_3), skip_special_tokens=False)
            # filled_summaries_4 += tokenizer.batch_decode(list(outputs_4), skip_special_tokens=False)
            
            input_articles += tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=False)
            #print(input_articles)
            #break
    filled_summaries = [x.replace('<pad>', '').replace('</s>', '').replace('<s>','').strip() for x in filled_summaries]
    #print(filled_summaries)
    filled_summaries_output = to_list(filled_summaries, num_return_sequences)
    #print(filled_summaries_output)
    filled_input = [x.replace('<pad>', '').replace('</s>', '').replace('<s>','').strip() for x in input_articles]
    
    # Initialize empty lists to store summary and article sentences
    summary_sentences = []
    article_sentences = []
    for data in filled_input:
      summary_start = data.find("summary:") + len("summary:")
      article_start = data.find("article:") + len("article:")
      summary = data[summary_start:article_start - len("article:")].strip()
      article = "(CNN)" + data[article_start:].strip()

      summary_sentences.append(summary)
      article_sentences.append(article)

    print(summary_sentences)
    print(article_sentences)
    
    # Save the summary and article to separate pickle files
    with open('summary_input.pickle', 'wb') as summary_file:
        pickle.dump(summary_sentences, summary_file)

    with open('article_input.pickle', 'wb') as article_file:
        pickle.dump(article_sentences, article_file)

    with open('summary_generated.pkl', 'wb') as f:
        pickle.dump(filled_summaries_output, f)
        
  

if __name__ == "__main__":
    main()
