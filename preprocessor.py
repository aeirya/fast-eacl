# device = 'mps'
device = 'cuda'

import os
from datetime import datetime
import json
import re
import torch
from transformers import BertTokenizer, BertModel
import numpy as np

# base_dir = r"C:\Users\AMIR\Desktop\stocknet_dataset_master\tweet\preprocessed"

tweets_base_dir = "stocknet-dataset/tweet/preprocessed"
base_dir = 'output'
# base_dir = r"C:\Users\AMIR\Desktop\stocknet_dataset_master\tweet\preprocessed"

source_dirs = [
    'AAPL', 'ABB', 'ABBV', 'AEP', 'AGFS', 'AMGN', 'AMZN', 'BA', 'BABA', 'BAC', 'BBL', 'BCH', 'BHP', 'BP', 
    'BRK-A', 'BSAC', 'BUD', 'C', 'CAT', 'CELG', 'CHL', 'CHTR', 'CMCSA', 'CODI', 'CSCO', 'CVX', 'D', 'DHR', 
    'DIS', 'DUK', 'EXC', 'FB', 'GD', 'GE', 'GOOG', 'HD', 'HON', 'HRG', 'HSBC', 'IEP', 'INTC', 'JNJ', 'JPM', 
    'KO', 'LMT', 'MA', 'MCD', 'MDT', 'MMM', 'MO', 'MRK', 'MSFT', 'NEE', 'NGG', 'NVS', 'ORCL', 'PCG', 'PCLN', 
    'PEP', 'PFE', 'PG', 'PICO', 'PM', 'PPL', 'PTR', 'RDS-B', 'REX', 'SLB', 'SNP', 'SNY', 'SO', 'SPLP', 'SRE', 
    'T', 'TM', 'TOT', 'TSM', 'UL', 'UN', 'UNH', 'UPS', 'UTX', 'V', 'VZ', 'WFC', 'WMT', 'XOM'
]
base_train_dir = r"C:\Users\AMIR\Desktop\Training"
base_test_dir = r"C:\Users\AMIR\Desktop\Test"
base_val_dir = r"C:\Users\AMIR\Desktop\Validation"
train_date_range = ('2014-01-01', '2015-07-31')
test_date_range = ('2015-10-01', '2016-01-01')
val_date_range = ('2015-08-01', '2015-09-30')
train_start_date, train_end_date = map(datetime.strptime, train_date_range, ['%Y-%m-%d']*2)
test_start_date, test_end_date = map(datetime.strptime, test_date_range, ['%Y-%m-%d']*2)
val_start_date, val_end_date = map(datetime.strptime, val_date_range, ['%Y-%m-%d']*2)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.to(device)

def get_destination_dir(base_dir, folder_name):
    return os.path.join(base_dir, folder_name)

def preprocess_tweet(text):
    text = re.sub(r'@\w+', '', text)  # Remove identifiers
    text = re.sub(r'#', '', text)     # Remove hashtags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    return text

def encode_tweet(text):
    if isinstance(text, list):
        text = [preprocess_tweet(t) for t in text]
    else:
        text = preprocess_tweet(text)

    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    
    # pass input to gpu
    for key, item in inputs.items():
        inputs[key] = item.to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        mask_sum = attention_mask.sum(dim=1).unsqueeze(-1)
        embeddings = (token_embeddings * attention_mask.unsqueeze(-1)).sum(dim=1) / mask_sum
        
        del inputs, outputs, token_embeddings, attention_mask
        return embeddings.squeeze().cpu().numpy()


def day_filepath_streamer(folder_name):
    # input: stock directory consisting of day files
    # output: input file name, output dir name, date str

    train_dir = get_destination_dir(base_train_dir, folder_name)
    test_dir = get_destination_dir(base_test_dir, folder_name)
    val_dir = get_destination_dir(base_val_dir, folder_name)
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    print(f"Processing directory: {folder_name}")
    for year in range(2014, 2018):
        for month in range(1, 13):
            for day in range(1, 32):
                try:
                    date_str = f"{year}-{month:02d}-{day:02d}"
                    date = datetime.strptime(date_str, '%Y-%m-%d')
                    if train_start_date.date() <= date.date() <= train_end_date.date():
                        dest_dir = train_dir
                    elif test_start_date.date() <= date.date() <= test_end_date.date():
                        dest_dir = test_dir
                    elif val_start_date.date() <= date.date() <= val_end_date.date():
                        dest_dir = val_dir
                    else:
                        continue
                    
                    file_path = os.path.join(tweets_base_dir, folder_name, date_str)
                    if not os.path.isfile(file_path):
                        continue
                
                    yield file_path, dest_dir, date_str

                except:
                    # print(f'error in listing files. date: {date_str}')
                    pass

def read_day_file(file_path):
    tweets = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                tweet_data = json.loads(line)
                
                # the tweet_data's text is an array of word stems(?)
                tweet_text = ' '.join(tweet_data['text'])
                
                # tweets date
                create_date = tweet_data['created_at']
                time = create_date.split()[3]
                time_format = "%H:%M:%S"
                # time of day in minutes
                time = datetime.strptime(time, time_format)

                tweets.append({
                    'text': tweet_text,
                    'time': time
                })
    except:
        print("error in reading files")
        raise
    
    tweets = sorted(tweets, key=lambda x: x["time"])
                    
    return tweets


def process_directory(folder_name):
    for file_path, dest_dir, date_str in day_filepath_streamer(folder_name):
        try:
            tweets = read_day_file(file_path)
            # encode text
            tweet_texts = [t['text'] for t in tweets]
            embeddings = encode_tweet(tweet_texts)
            embeddings_path = os.path.join(dest_dir, f"{date_str}_embeddings.npy")
            np.save(embeddings_path, np.array(embeddings))
            print(f"Saved embeddings to {embeddings_path}")

            # create timestamp data
            timestamps = [1.0]
            for t1, t2 in zip(tweets[:-1], tweets[1:]):
                deltaT = (t2['time'] - t1['time']).total_seconds()/60
                timestamps.append(deltaT)

        except:
            print("error in encoding file")
            raise
            continue


def process_directory_batched(folder_name):
    # aggregated tweets
    all_tweets = []

    # saved info for each file
    stack = []

    file_idx = 0
    for file_path, dest_dir, date_str in day_filepath_streamer(folder_name):
        tweet_texts = read_day_file(file_path)
        all_tweets += tweet_texts

        count = len(tweet_texts)
        stack.append(
            (count, dest_dir, date_str)
        )

        # file_idx += 1
        # if file_idx > 20:
        #     break

    # encode all the tweets at the same time
    embeddings = []
    step = 20
    for i in range(0, len(all_tweets), step):
        embeddings.append(encode_tweet(all_tweets[i:i+step]))
    embeddings = np.vstack(embeddings)


    # encoded tweets based on day
    # days = {}

    i = 0
    for tweet_count, dest_dir, date_str in stack:
        e = embeddings[i:i+tweet_count]
        i += tweet_count

        embeddings_path = os.path.join(dest_dir, f"{date_str}_embeddings.npy")
        np.save(embeddings_path, np.array(e))
        print(f"Saved embeddings to {embeddings_path}")

    del embeddings





for folder_name in source_dirs:
    process_directory(folder_name)

print("Processing complete.")
