from utils import *
from dataset import BilingualDataset, triangular_mask
from model import transformer

import os
import torch
import torch.nn as nn
from tqdm import tqdm
import pickle
from torch.utils.data import random_split, DataLoader
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase, Sequence, Strip
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from datasets import load_dataset, get_dataset_split_names, get_dataset_config_names


def validation_step(config,model,dataset,device):
        model.eval()
        batch = next(iter(dataset['val_dataset']))
        # print('val check 101')                           
        sos_idx = dataset['tgt_tokenizer'].token_to_id('[SOS]')
        eos_idx = dataset['tgt_tokenizer'].token_to_id('[EOS]')

        encoder_input = batch['encoder_input'].to(device) # (batch, seq_len)
        decoder_input = batch['decoder_input'].to(device) # (batch, seq_len)
        encoder_mask = batch['encoder_mask'].to(device) # (1,1,seq_len)
        decoder_mask = batch['decoder_mask'].to(device) # (1,seq_len,seq_len)

        with torch.no_grad():
                if torch.cuda.device_count() > 1:        
                        encoder_output = model.module.encoder(encoder_input, encoder_mask) # (batch, seq_len, d_model)
                else:
                        encoder_output = model.encoder(encoder_input, encoder_mask) # (batch, seq_len, d_model)
                
                decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(encoder_input).to(device)

                while True:
                        if decoder_input.size(1) == config['tgt_seq_len']:
                                break
                        # build mask for target
                        decoder_mask = triangular_mask(decoder_input.size(1)).type_as(encoder_mask).to(device)

                        # calculate output
                        if torch.cuda.device_count() > 1:
                                out = model.module.decoder(decoder_input, encoder_output, encoder_mask, decoder_mask) # (batch, seq_len, d_model)
                        else:
                                out = model.decoder(decoder_input, encoder_output, encoder_mask, decoder_mask) # (batch, seq_len, d_model)

                        # get next token
                        if torch.cuda.device_count() > 1:
                                prob = model.projection(out[:, -1]) # (batch, seq_len, vocab_size)
                        else:
                                prob = model.module.projection(out[:, -1]) # (batch, seq_len, vocab_size)
                        _, next_word = torch.max(prob, dim=1)
                        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(encoder_input).fill_(next_word.item()).to(device)], dim=1)

                        if next_word.item() == eos_idx:
                                break

        source_text = batch["src_text"][0]
        target_text = batch["tgt_text"][0]
        model_out_text = dataset['tgt_tokenizer'].decode(decoder_input.squeeze(0).detach().cpu().numpy())
        print(f">>>> Actual sentence in '{config['src_lang']}':", source_text)
        print(f">>>> Actual sentence in '{config['tgt_lang']}':", target_text)
        print(f">>>> Translated sentence from '{config['src_lang']}' to '{config['tgt_lang']}':", model_out_text)
                # break
        return 

def get_data(dataset, language):
        for data in dataset:
                yield data['translation'][language] 

def get_tokenizer(config, lang, dataset, lower=True):
        path = os.path.join(os.getcwd(),config['data_path'],config['tokenizer_file'])
        if not os.path.exists(path.format(lang=lang)):
                print(f'\nGenerating text vocabulary of {lang} language..')
                tm = Time()
                tm.start('generating vocabulary')
                print('>> The Hugging-Face dataset used:',config['dataset_name'])

                # dataset_configs = get_dataset_config_names(config['dataset_name'])
                # print('>> Avilable translations from English:',[x for x in dataset_configs if x[:2]==lang])

                # If do not have internet connection while running code, comment out the below line 
                print('>> Avilable splits in the dataset:',get_dataset_split_names(config['dataset_name'], config['dataset_config_name']))
                
                tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
                if lower:
                        tokenizer.normalizer =  Sequence([Strip(), Lowercase()])
                else:
                        tokenizer.normalizer = Strip()
                tokenizer.pre_tokenizer = Whitespace()
                trainer = WordLevelTrainer(show_progress=True, min_frequency=2, special_tokens=['[UNK]','[PAD]','[SOS]','[EOS]'])
                tokenizer.train_from_iterator(get_data(dataset, lang), trainer, length=dataset.num_rows)
                tokenizer.save(path.format(lang=lang), pretty=True)
                tm.end()
        else:
                print(f'>> Tokenizer for "{lang}" found at {path.format(lang=lang)}.')
                tokenizer = Tokenizer.from_file(path.format(lang=lang))
        return tokenizer

def get_dataset(config):
        print('\nDownloading dataset..')
        data_path = Path(os.path.join(os.getcwd(), config['data_path'], config['dataset_name']+'.pkl'))
        tm = Time()
        if data_path.exists():
                print(f'>> Data file found at {data_path}')
                print(">> Loading the data file.")
                with open(data_path, 'rb') as file:
                        dataset = pickle.load(file)
        else:
                print(f">> Data file not found. Downloading and saving data..")
                tm.start('download data')
                dataset = load_dataset(config['dataset_name'], config['dataset_config_name'], split=['train'], cache_dir=os.path.join(os.getcwd(), 'data'))[0]
                print('>> Number of translation samples:', dataset.num_rows)                
                # Save the data to a pickle file
                with open(data_path, 'wb') as file:
                        pickle.dump(dataset, file)
                print(f">> Data saved to pickle file: {data_path}")
                tm.end()

        src_tokenizer = get_tokenizer(config, config['src_lang'], dataset)
        tgt_tokenizer = get_tokenizer(config, config['tgt_lang'], dataset, lower=False)
        
        print('\nLoading dataset..')
        tm.start('loading data')
        src_max_seq_len, tgt_max_seq_len = 0,0
        for data in dataset:
                src_seq_len = len(src_tokenizer.encode(data['translation'][config['src_lang']]))
                tgt_seq_len = len(tgt_tokenizer.encode(data['translation'][config['tgt_lang']]))
                src_max_seq_len = max(src_max_seq_len, src_seq_len)
                tgt_max_seq_len = max(tgt_max_seq_len, tgt_seq_len)
        print(f'>> Maximum sequence length of\n\t1. source language "{config["src_lang"]}": {src_max_seq_len}\n\t2. target language "{config["tgt_lang"]}": {tgt_max_seq_len}')
        config['src_seq_len'] = src_max_seq_len
        config['tgt_seq_len'] = tgt_max_seq_len

        train_size = int(config['dataset_split_ratio']*dataset.num_rows)
        val_size = dataset.num_rows-train_size
        train_dataset_raw, val_dataset_raw = random_split(dataset, lengths=[train_size, val_size])

        train_dataset = BilingualDataset(train_dataset_raw, src_tokenizer, tgt_tokenizer, config['src_lang'], config['tgt_lang'], src_max_seq_len, tgt_max_seq_len)
        val_dataset = BilingualDataset(val_dataset_raw, src_tokenizer, tgt_tokenizer, config['src_lang'], config['tgt_lang'], src_max_seq_len, tgt_max_seq_len)

        train_dataset = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_dataset = DataLoader(val_dataset, batch_size=1, shuffle=True)
        tm.end()
        return {
                'train_dataset': train_dataset,
                'val_dataset': val_dataset,
                'src_tokenizer': src_tokenizer,
                'tgt_tokenizer': tgt_tokenizer
        }

def train_model(config):        
        Path(f"{(config['dataset_config_name'])}-{config['model_folder']}").mkdir(parents=True, exist_ok=True)
        dataset = get_dataset(config)
        print('\Setting up hardwares..')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = transformer(src_seq_len = config['src_seq_len'],
                            tgt_seq_len = config['tgt_seq_len'],
                            src_vocab_size = dataset['src_tokenizer'].get_vocab_size(),
                            tgt_vocab_size = dataset['tgt_tokenizer'].get_vocab_size(),
                            N = config['N'],
                            d_model = config['d_model'],
                            head = config['head'],
                            d_ff = config['d_ff'],
                            dropout = config['dropout']
                        ).to(device)
        
        # change model configuration if multiple GPU avialable
        if torch.cuda.device_count() > 1:
                print(">> Using", torch.cuda.device_count(), "GPUs!")
                model = nn.DataParallel(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
        loss_function = torch.nn.CrossEntropyLoss(ignore_index=dataset['tgt_tokenizer'].token_to_id('[PAD]'), label_smoothing=config['label_smoothing']).to(device)

        model_file = latest_weights_file_path(config) if config['preload']=='latest' else get_weights_file_path(config, config['preload']) if config['preload'] else None
        
        init_epoch = 0
        print(f'\n\nTraining model on {device}..')
        if model_file:
                state = torch.load(model_file)
                init_epoch = state['epoch']+1
                print(f'>> Resuming model training from epoch no. {init_epoch}')
                model.load_state_dict(state['model'])
                optimizer.load_state_dict(state['optimizer'])
        else:
                print('>> No model to preload, starting from scratch.')
        
        ttrain = Time()
        ttrain.start('training model')
        tepoch, tbatch = Time(), Time()
        for epoch in range(init_epoch, config['epochs']):
                torch.cuda.empty_cache()
                model.train()
                tepoch.start()

                # training step
                ## batch_iterator = tqdm(dataset['train_dataset'], desc=f'>> Processing epoch {epoch:03d}') 
                ## for batch in batch_iterator:
                batch_count = 1
                print(f'\n>> epoch: {epoch}')
                for batch in dataset['train_dataset']:
                        tbatch.start('>>> @batch-{batch_count} , @epoch-{epoch}, loss-{loss}')
                        encoder_input = batch['encoder_input'].to(device) # (batch, seq_len)
                        decoder_input = batch['decoder_input'].to(device) # (batch, seq_len)
                        encoder_mask = batch['encoder_mask'].to(device) # (1,1,seq_len)
                        decoder_mask = batch['decoder_mask'].to(device) # (1,seq_len,seq_len)
                
                        if torch.cuda.device_count() > 1:
                                encoder_output = model.module.encoder(encoder_input, encoder_mask) # (batch, seq_len, d_model)
                                decoder_output = model.module.decoder(decoder_input, encoder_output, encoder_mask, decoder_mask) # (batch, seq_len, d_model)
                                output = model.module.projection(decoder_output) # (batch, seq_len, vocab_size)
                        else:
                                encoder_output = model.encoder(encoder_input, encoder_mask) # (batch, seq_len, d_model)
                                decoder_output = model.decoder(decoder_input, encoder_output, encoder_mask, decoder_mask) # (batch, seq_len, d_model)
                                output = model.projection(decoder_output) # (batch, seq_len, vocab_size)
                        
                        label = batch['label'].to(device) # (batch, seq_len)
                        loss = loss_function(output.contiguous().view(-1, dataset['tgt_tokenizer'].get_vocab_size()), label.view(-1))
                        # batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
                        
                        # back propagation step
                        loss.backward()

                        # update parameters using optimizer
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
        
                        tbatch.message = tbatch.message.format(batch_count=batch_count, loss=f'{loss.item():6.3f}', epoch=epoch)
                        tbatch.end()
                        # if batch_count==3:
                        #         break

                        batch_count += 1

                # save the model instance at end of every epoch
                file = get_weights_file_path(config,f'{epoch:03d}')
                torch.save({
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                }, f = file)

                # validation step
                if config['validation_step_while_training']:
                        if epoch % config['validation_step_frequency']==0:
                                print('>>\n>> Validation step..')
                                tval = Time()
                                tval.start(f'validation step @epoch-{epoch}')
                                validation_step(config,model,dataset,device)
                                tval.end()
                tepoch.end()
        print()
        ttrain.end()        
        return
 
def main(config):
        train_model(config)
        
if __name__=='__main__':
        config = load_config('config.yml')
        main(config)

