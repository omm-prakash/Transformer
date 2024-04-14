import torch
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
        def __init__(self, raw_dataset, src_tokenizer, tgt_tokenizer, src_lang, tgt_lang, src_seq_len, tgt_seq_len) -> None:
                super().__init__()
                self.raw_dataset = raw_dataset
                self.src_tokenizer = src_tokenizer
                self.tgt_tokenizer = tgt_tokenizer
                self.src_lang = src_lang
                self.tgt_lang = tgt_lang
                self.src_seq_len = src_seq_len
                self.tgt_seq_len = tgt_seq_len

                self.encoder_input_format = '[SOS] {sen} [EOS]'
                self.decoder_input_format = '[SOS] {sen}'
                self.output_format = '{sen} [EOS]'
                self.pad_token = src_tokenizer.token_to_id('[PAD]')
        
        def __len__(self):
                return len(self.raw_dataset)
        
        def __getitem__(self, index):
                sentence = self.raw_dataset[index]
                encoder_input_sentence = self.encoder_input_format.format(sen = sentence['translation'][self.src_lang])
                decoder_input_sentence = self.decoder_input_format.format(sen = sentence['translation'][self.tgt_lang])
                output_sentence = self.output_format.format(sen = sentence['translation'][self.tgt_lang])

                # encoder input padding
                encoder_input = self.src_tokenizer.encode(encoder_input_sentence)
                encoder_input.pad(pad_id=self.pad_token, length=self.src_seq_len)
                encoder_input = torch.tensor(encoder_input.ids, dtype=torch.int64)

                # decoder input padding
                decoder_input = self.tgt_tokenizer.encode(decoder_input_sentence)
                decoder_input.pad(pad_id=self.pad_token, length=self.tgt_seq_len)
                decoder_input = torch.tensor(decoder_input.ids, dtype=torch.int64)

                # label padding
                label = self.tgt_tokenizer.encode(output_sentence)
                label.pad(pad_id=self.pad_token, length=self.tgt_seq_len)
                label = torch.tensor(label.ids, dtype=torch.int64)

                assert encoder_input.size(0)==self.src_seq_len
                assert decoder_input.size(0)==self.tgt_seq_len
                assert label.size(0)==self.tgt_seq_len

                return {
                        'encoder_input': encoder_input, 
                        'decoder_input': decoder_input,
                        'label': label,
                        'encoder_mask': (encoder_input!=self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1,1,seq_len)
                        'decoder_mask': (decoder_input!=self.pad_token).unsqueeze(0).unsqueeze(0).int() & triangular_mask(self.tgt_seq_len), # (1,1,seq_len) & (1,seq_len,seq_len)
                        'src_text': sentence['translation'][self.src_lang],
                        'tgt_text': sentence['translation'][self.tgt_lang]
                }
        
        
def triangular_mask(size):
        mask = torch.triu(torch.ones(1,size,size), diagonal=1)
        return mask==0