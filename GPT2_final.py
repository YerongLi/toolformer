# coding: utf-8

# In[ ]:


# get_ipython().system('pip install transformers')


# In[ ]:


import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import torch.nn.functional as F
import csv


# # GPT2 with Fine Tuning

# ### Prepare data

# In[ ]:


lyrics = pd.read_csv('../data/lyrics-data.csv')
print('head')
print(lyrics.head())
lyrics = lyrics[lyrics['Idiom']=='ENGLISH']


# In[ ]:


#Only keep popular artists, with genre Rock/Pop and popularity high enough
artists = pd.read_csv('/content/drive/MyDrive/Google colabs/artists-data.csv')

artists = artists[(artists['Genre'].isin(['Rock'])) & (artists['Popularity']>5)]


# In[ ]:


df = lyrics.merge(artists[['Artist', 'Genre', 'Link']], left_on='ALink', right_on='Link', how='inner')


# In[ ]:


df = df.drop(columns=['ALink','SLink','Idiom','Link'])


# In[ ]:


#Drop the songs with lyrics too long (after more than 1024 tokens, does not work)
df = df[df['Lyric'].apply(lambda x: len(x.split(' ')) < 350)]


# In[ ]:


len(df)


# In[ ]:


#Create a very small test set to compare generated text with the reality
test_set = df.sample(n = 500)
df = df.loc[~df.index.isin(test_set.index)]

#Reset the indexes
test_set = test_set.reset_index()
df = df.reset_index()


# In[ ]:


#For the test set only, keep last 20 words in a new column, then remove them from original column
test_set['True_end_lyrics'] = test_set['Lyric'].str.split().str[-20:].apply(' '.join)
test_set['Lyric'] = test_set['Lyric'].str.split().str[:-20].apply(' '.join)


# In[ ]:


test_set.head()


# ### Prepare the dataset

# In[ ]:


class SongLyrics(Dataset):
    
    def __init__(self, control_code, truncate=False, gpt2_type="gpt2", max_length=1024):

        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.lyrics = []

        for row in df['Lyric']:
          self.lyrics.append(torch.tensor(
                self.tokenizer.encode(f"<|{control_code}|>{row[:max_length]}<|endoftext|>")
            ))
                
        if truncate:
            self.lyrics = self.lyrics[:20000]
        self.lyrics_count = len(self.lyrics)
        
    def __len__(self):
        return self.lyrics_count

    def __getitem__(self, item):
        return self.lyrics[item]


# In[ ]:


dataset = SongLyrics(df['Lyric'], truncate=True, gpt2_type="gpt2")


# ### Prepare training

# In[ ]:


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')


# In[ ]:


#Accumulated batch size (since GPT2 is so big)
def pack_tensor(new_tensor, packed_tensor, max_seq_len):
    if packed_tensor is None:
        return new_tensor, True, None
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
        return packed_tensor, True, None


# In[ ]:


def train(
    dataset, model, tokenizer,
    batch_size=16, epochs=20, lr=2e-5,
    max_seq_len=400, warmup_steps=200,
    gpt2_type="gpt2", output_dir=".", output_prefix="wreckgar",
    test_mode=False,save_model_on_epoch=False,
):

    acc_steps = 100
    device=torch.device("cuda")
    model = model.cuda()
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
    )

    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    loss=0
    accumulating_batch_count = 0
    input_tensor = None

    for epoch in range(epochs):

        print(f"Training epoch {epoch}")
        print(loss)
        for idx, entry in tqdm(enumerate(train_dataloader)):
            (input_tensor, carry_on, remainder) = pack_tensor(entry, input_tensor, 768)

            if carry_on and idx != len(train_dataloader) - 1:
                continue

            input_tensor = input_tensor.to(device)
            outputs = model(input_tensor, labels=input_tensor)
            loss = outputs[0]
            loss.backward()

            if (accumulating_batch_count % batch_size) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            accumulating_batch_count += 1
            input_tensor = None
        if save_model_on_epoch:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch}.pt"),
            )
    return model


# ### Actual Training

# In[ ]:


#Train the model on the specific data we have
model = train(dataset, model, tokenizer)


# In[ ]:


#Save the model to a pkl or something so it can be reused later on
torch.save(model, '/content/drive/MyDrive/Google colabs/model.pt')


# ### Text generation

# In[ ]:


#Load the model to use it
model = torch.load('/content/drive/MyDrive/Google colabs/model.pt')


# In[ ]:


def generate(
    model,
    tokenizer,
    prompt,
    entry_count=10,
    entry_length=30, #maximum number of words
    top_p=0.8,
    temperature=1.,
):

    model.eval()

    generated_num = 0
    generated_list = []

    filter_value = -float("Inf")

    with torch.no_grad():

        for entry_idx in trange(entry_count):

            entry_finished = False

            generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

            for i in range(entry_length):
                outputs = model(generated, labels=generated)
                loss, logits = outputs[:2]
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value

                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)

                if next_token in tokenizer.encode("<|endoftext|>"):
                    entry_finished = True

                if entry_finished:

                    generated_num = generated_num + 1

                    output_list = list(generated.squeeze().numpy())
                    output_text = tokenizer.decode(output_list)
                    generated_list.append(output_text)
                    break
            
            if not entry_finished:
              output_list = list(generated.squeeze().numpy())
              output_text = f"{tokenizer.decode(output_list)}<|endoftext|>" 
              generated_list.append(output_text)
                
    return generated_list


# In[ ]:


#Function to generate multiple sentences. Test data should be a dataframe
def text_generation(test_data):
  generated_lyrics = []
  for i in range(len(test_data)):
    x = generate(model.to('cpu'), tokenizer, test_data['Lyric'][i], entry_count=1)
    generated_lyrics.append(x)
  return generated_lyrics


# In[ ]:


generated_lyrics = text_generation(test_set)


# In[ ]:


#Loop to keep only generated text and add it as a new column in the dataframe
my_generations=[]

for i in range(len(generated_lyrics)):
  a = test_set['Lyric'][i].split()[-30:] #Get the matching string we want (30 words)
  b = ' '.join(a)
  c = ' '.join(generated_lyrics[i]) #Get all that comes after the matching string
  my_generations.append(c.split(b)[-1])

test_set['Generated_lyrics'] = my_generations


# In[ ]:


#Finish the sentences when there is a point, remove after that
final=[]

for i in range(len(test_set)):
  to_remove = test_set['Generated_lyrics'][i].split('.')[-1]
  final.append(test_set['Generated_lyrics'][i].replace(to_remove,''))

test_set['Generated_lyrics'] = final
test_set.head()


# In[ ]:


test_set['Generated_lyrics'][7]


# In[ ]:


test_set['True_end_lyrics'][7]


# ### Analyze performance

# In[ ]:


#Using BLEU score to compare the real sentences with the generated ones
import statistics
from nltk.translate.bleu_score import sentence_bleu

scores=[]

for i in range(len(test_set)):
  reference = test_set['True_end_lyrics'][i]
  candidate = test_set['Generated_lyrics'][i]
  scores.append(sentence_bleu(reference, candidate))

statistics.mean(scores)


# In[ ]:


#Rouge score
from rouge import Rouge
rouge=Rouge()

rouge.get_scores(test_set['Generated_lyrics'], test_set['True_end_lyrics'], avg=True)


# # GPT2 without any fine Tuning

# In[ ]:


import transformers
import torch


# In[ ]:


tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
model = transformers.GPT2LMHeadModel.from_pretrained('gpt2')


# In[ ]:


## Making a function that will generate text for us ##
def gen_text(prompt_text, tokenizer, model, n_seqs=1, max_length=374):
  # n_seqs is the number of sequences to generate
  # max_length is the maximum length of the sequence
  encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
  # We are encoding the text using the gpt tokenizer. The return tensors are of type "pt"
  # since we are using PyTorch, not tensorflow
  output_sequences = model.generate(
      input_ids=encoded_prompt,
      max_length=max_length+len(encoded_prompt), # The model has to generate something, 
      # so we add the length of the original sequence to max_length
      temperature=1.0,
      top_k=0,
      top_p=0.9,
      repetition_penalty=1.2, # To ensure that we dont get repeated phrases
      do_sample=True,
      num_return_sequences=n_seqs
  ) # We feed the encoded input into the model.
  ## Getting the output ##
  if len(output_sequences.shape) > 2:
    output_sequences.squeeze_() # the _ indicates that the operation will be done in-place
  generated_sequences = []
  for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
    generated_sequence = generated_sequence.tolist()
    text = tokenizer.decode(generated_sequence)
    total_sequence = (
        prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True, )) :]
    )
    generated_sequences.append(total_sequence)
  return generated_sequences


# In[ ]:


#Generate sequences
gen_text(df['Lyric'][0],tokenizer,model)


# In[ ]:


#Function to generate multiple sentences. Test data should be a dataframe
def text_generation(test_data):
  generated_lyrics = []
  for i in range(len(test_data)):
    x = gen_text(test_data['Lyric'][i], tokenizer, model)
    generated_lyrics.append(x)
  return generated_lyrics

generated_lyrics = text_generation(test_set)


# In[ ]:


#Loop to keep only generated text and add it as a new column in the dataframe
my_generations=[]

for i in range(len(generated_lyrics)):
  a = test_set['Lyric'][i].split()[-30:] #Get the matching string we want (30 words)
  b = ' '.join(a)
  c = ' '.join(generated_lyrics[i]) #Get all that comes after the matching string
  my_generations.append(c.split(b)[-1])

test_set['Generated_lyrics'] = my_generations


# In[ ]:


#Finish the sentences when there is a point, remove after that
final=[]

for i in range(len(test_set)):
  to_remove = test_set['Generated_lyrics'][i].split('.')[-1]
  final.append(test_set['Generated_lyrics'][i].replace(to_remove,''))

test_set['Generated_lyrics'] = final
test_set.head()


# In[ ]:


#Using BLEU score to compare the real sentences with the generated ones
import statistics
from nltk.translate.bleu_score import sentence_bleu

scores=[]

for i in range(len(test_set)):
  reference = test_set['True_end_lyrics'][i]
  candidate = test_set['Generated_lyrics'][i]
  scores.append(sentence_bleu(reference, candidate))

statistics.mean(scores)


# In[ ]:


get_ipython().system('pip install rouge')


# In[ ]:


#Rouge score
from rouge import Rouge
rouge=Rouge()

rouge.get_scores(test_set['Generated_lyrics'], test_set['True_end_lyrics'], avg=True, ignore_empty=True)

