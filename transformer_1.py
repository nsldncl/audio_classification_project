import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer,TransformerDecoder, TransformerDecoderLayer

import csv
import pickle
import numpy as np
from pathlib import Path
import yaml
from dotmap import DotMap
import time
import sys
from loguru import logger
from tqdm import tqdm
from pathlib import Path
import math
from pprint import PrettyPrinter
from warmup_scheduler import GradualWarmupScheduler


#defining custom functions

#csv file contain captions
def write_csv_file(csv_obj, file_name):
    with open(file_name, 'w') as f:
        writer = csv.DictWriter(f, csv_obj[0].keys())
        writer.writeheader()
        writer.writerows(csv_obj)
    print(f'Write to {file_name} successfully.')

def load_csv_file(file_name):
    with open(file_name, 'r') as f:
        csv_reader = csv.DictReader(f)
        csv_obj = [csv_line for csv_line in csv_reader]
    return csv_obj

#pickle file contains wordlist

def load_picke_file(file_name):
    with open(file_name, 'rb') as f:
        pickle_obj = pickle.load(f)
    return pickle_obj

def write_pickle_file(obj, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)
    print(f'Write to {file_name} successfully.')

#settings.yaml file contains arguments that are used to train the model 

def get_config():
    with open('settings.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = DotMap(config)
    return config
    
    
#Creating dataset class
#for data loader
class ClothoDataset(Dataset):
    def __init__(self, split, #data loader function 
                 input_field_name,
                 load_into_memory):
        super(ClothoDataset, self).__init__()
        split_dir = Path('data/data_splits', split)
        self.examples = sorted(split_dir.iterdir()) #we have to convert the data into batches 
        self.input_field_name = input_field_name
        self.output_field_name = 'words_indexs'
        self.load_into_memory = load_into_memory
        if load_into_memory:
            self.examples = [np.load(str(file), allow_pickle=True) for file in self.examples]

    def __len__(self):
        return len(self.examples) #how much data we have

    def __getitem__(self, index): #return five data points
        item = self.examples[index]
        if not self.load_into_memory:
            item = np.load(str(item), allow_pickle=True)
        feature = item[self.input_field_name].item()  # waveform or log melspectorgram
        words_indexs = item[self.output_field_name].item()
        file_name = str(item['file_name'].item())
        caption_len = len(words_indexs)
        caption = str(item['caption'].item())
        return feature, words_indexs, file_name, caption_len, caption

#Dataloader function this will use the above class
#dataloader means converting data in batches
def get_clotho_loader(split,
                      input_field_name,
                      load_into_memory,
                      batch_size,
                      shuffle=False,
                      drop_last=False,
                      num_workers=1):
    dataset = ClothoDataset(split, input_field_name, load_into_memory)
    if input_field_name == 'audio_data': #if it is audio then call audio one
        return DataLoader(dataset=dataset, batch_size=batch_size,
                          shuffle=shuffle, drop_last=drop_last,
                          num_workers=num_workers, collate_fn=clotho_collate_fn_audio)
    else:
        return DataLoader(dataset=dataset, batch_size=batch_size,
                          shuffle=shuffle, drop_last=drop_last,
                          num_workers=num_workers, collate_fn=clotho_collate_fn)

#converting npy file into batches
#if we pass a dataset, then firstly we will find maximum value of audio or caption, 
#then we have to change the length of all the captions or audio to same length as of max value
#then we will apply padding for filling up the null values

def clotho_collate_fn(batch): 
    max_feature_time_steps = max(i[0].shape[0] for i in batch)
    max_caption_length = max(i[1].shape[0] for i in batch)
    feature_number = batch[0][0].shape[-1]
    eos_token = batch[0][1][-1]
    feature_tensor, words_tensor = [], []
    for feature, words_indexs, _, _, _ in batch:
        if max_feature_time_steps > feature.shape[0]:
            padding = torch.zeros(max_feature_time_steps - feature.shape[0], feature_number).float()
            data = [torch.from_numpy(feature).float()]
            data.append(padding)
            temp_feature = torch.cat(data)
        else:
            temp_feature = torch.from_numpy(feature[:max_feature_time_steps, :]).float()
        feature_tensor.append(temp_feature.unsqueeze_(0))

        if max_caption_length > words_indexs.shape[0]:
            padding = torch.ones(max_caption_length - len(words_indexs)).mul(eos_token).long()
            data = [torch.from_numpy(words_indexs).long()]
            data.append(padding)
            tmp_words_indexs = torch.cat(data)
        else:
            tmp_words_indexs = torch.from_numpy(words_indexs[:max_caption_length]).long()
        words_tensor.append(tmp_words_indexs.unsqueeze_(0))
    feature_tensor = torch.cat(feature_tensor)
    words_tensor = torch.cat(words_tensor)
    file_names = [i[2] for i in batch]
    caption_lens = [i[3] for i in batch]
    captions = [i[4] for i in batch]
    return feature_tensor, words_tensor, file_names, caption_lens, captions

#converting audio files into batches
def clotho_collate_fn_audio(batch):
    max_audio_time_steps = max(i[0].shape[0] for i in batch)
    max_caption_length = max(i[1].shape[0] for i in batch)
    eos_token = batch[0][1][-1]
    audio_tensor, words_tensor = [], []
    for audio, words_indexs, _, _, _ in batch:
        if max_audio_time_steps >= audio.shape[0]:
            padding = torch.zeros(max_audio_time_steps - audio.shape[0]).float()
            data = [torch.from_numpy(audio).float()]
            data.append(padding)
            temp_audio = torch.cat(data)
        else:
            temp_audio = torch.from_numpy(audio[:max_audio_time_steps]).float()
        audio_tensor.append(temp_audio.unsqueeze_(0))
        if max_caption_length >= words_indexs.shape[0]:
            padding = torch.ones(max_caption_length - len(words_indexs)).mul(eos_token).long()
            data = [torch.from_numpy(words_indexs).long()]
            data.append(padding)
            tmp_words_indexs = torch.cat(data)
        else:
            tmp_words_indexs = torch.from_numpy(words_indexs[:max_caption_length]).long()
        words_tensor.append(tmp_words_indexs.unsqueeze_(0))
    audio_tensor = torch.cat(audio_tensor)
    words_tensor = torch.cat(words_tensor)
    file_names = [i[2] for i in batch]
    caption_lens = [i[3] for i in batch]
    captions = [i[4] for i in batch]
    return audio_tensor, words_tensor, file_names, caption_lens, captions


#just for printing logger data progress
#we are padding and masking our target captions

def rotation_logger(x, y): #print the process or information of task
    """Callable to determine the rotation of files in logger.
    :param x: Str to be logged.
    :type x: loguru._handler.StrRecord
    :param y: File used for logging.
    :type y: _io.TextIOWrapper
    :return: Shall we switch to a new file?
    :rtype: bool
    """
    return 'Captions start' in x

#masking the tokens
def set_tgt_padding_mask(tgt, tgt_len):
    batch_size = tgt.shape[0]
    max_len = tgt.shape[1]
    mask = torch.zeros(tgt.shape).type_as(tgt).to(tgt.device)
    for i in range(batch_size):
        num_pad = max_len - tgt_len[i]
        mask[i][max_len - num_pad:] = 1
    mask = mask.float().masked_fill(mask == 1, True).masked_fill(mask == 0, False).bool()
    return mask

#done in previous session
#explained
#it will give features from the audio, then we will pass the features, captions, bundles, etc. to the transformer
#transformer will encode the captions and then decode the outputs

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
def init_bn(bn):
    """Initialize a BatchNorm layer."""
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.max_pool2d(x, kernel_size=pool_size)
            x2 = F.avg_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x

class Cnn10(nn.Module):
    def __init__(self, config):
        super(Cnn10, self).__init__()
        self.input_data = config.data.input_field_name
        self.bn0 = nn.BatchNorm2d(64)
        if self.input_data == 'audio_data':
            sr = config.wave.sr
            window_size = config.wave.window_size
            hop_length = config.wave.hop_length
            mel_bins = config.wave.mel_bins
            fmin = config.wave.fmin
            fmax = config.wave.fmax
            self.spectrogram_extractor = Spectrogram(n_fft=window_size,
                                                     hop_length=hop_length,
                                                     win_length=window_size,
                                                     window='hann',
                                                     center=True,
                                                     pad_mode='reflect',
                                                     freeze_parameters=True)
            self.logmel_extractor = LogmelFilterBank(sr=sr, n_fft=window_size,
                                                     n_mels=mel_bins,
                                                     fmin=fmin,
                                                     fmax=fmax,
                                                     ref=1.0,
                                                     amin=1e-10,
                                                     top_db=None,
                                                     freeze_parameters=True)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.fc1 = nn.Linear(512, 512, bias=True)
        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.fc1)

    def forward(self, input, mixup_param=None):
        """ input: (batch_size, time_steps, mel_bins)"""
        if self.input_data == 'audio_data':
            x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
            x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        else:
            x = input.unsqueeze(1)  # (batch_size, 1, time_steps, mel_bins)
        if mixup_param is not None:
            lam, index = mixup_param
            x = lam * x + (1 - lam) * x[index]
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)  # average in the frequency domain (batch_size, channel, time)
        x = x.permute(2, 0, 1)  # time x batch x channel (512)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        return x
        
        
def init_layer(layer):
  """ Initialize a Linear or Convolutional layer. """
  nn.init.xavier_uniform_(layer.weight)
  if hasattr(layer, 'bias'):
      if layer.bias is not None:
          layer.bias.data.fill_(0.)

class AudioLinear(nn.Module):
  def __init__(self, nhid):
      super(AudioLinear, self).__init__()
      # self.fc1 = nn.Linear(512, 512, bias=True)
      self.fc2 = nn.Linear(512, nhid, bias=True)
      self.init_weights()

  def init_weights(self):
      # init_layer(self.fc1)
      init_layer(self.fc2)

  def forward(self, x):
      x = F.relu_(self.fc2(x))  # time x batch x nhid
      return x
      
      
#In this transfomer all the data are pass directly so we need to mention the position of each tokens.
#it will define the position for the tokens
#Transformer is multi headed 

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=2000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)




#transformer model begins here.........................................................................
class TransformerModel(nn.Module):
    """ Container module with an Cnn encoder and a Transformer decoder."""

    def __init__(self, config, words_list, pretrained_cnn=None): 
        super(TransformerModel, self).__init__()
        self.model_type = 'Cnn+Transformer'
        ntoken = len(words_list)
        # setting for CNN
        self.feature_extractor = Cnn10(config)
        #if our model is not pretrained then remove external data information
        if pretrained_cnn is not None:
            final = pretrained_cnn["model"] #removing the external data
            final.pop("fc_audioset.weight") 
            final.pop("fc_audioset.bias")
            self.feature_extractor.load_state_dict(final)
        #do not train if pretrained model
        if config.encoder.freeze:
            for name, p in self.feature_extractor.named_parameters():
                p.requires_grad = False

        # decoder settings
        self.decoder_only = config.decoder.decoder_only
        nhead = config.decoder.nhead            # number of heads in Transformer
        self.nhid = config.decoder.nhid         # number of expected features in decoder inputs
        nlayers = config.decoder.nlayers        # number of sub-decoder-layer in the decoder
        dim_feedforward = config.decoder.dim_feedforward   # dimension of the feedforward model
        activation = config.decoder.activation  # activation function of decoder intermediate layer
        dropout = config.decoder.dropout        # the dropout value

        #define the positional and audio linear encoder
        self.pos_encoder = PositionalEncoding(self.nhid, dropout)
        self.audio_linear = AudioLinear(self.nhid)
  
        ''' Including transfomer encoder '''
        encoder_layers = TransformerEncoderLayer(self.nhid,
                                                  nhead,
                                                  dim_feedforward,
                                                  dropout,
                                                  activation)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        ''' Including Transformer Decoder '''
        decoder_layers = TransformerDecoderLayer(self.nhid,
                                                 nhead,
                                                 dim_feedforward,
                                                 dropout,
                                                 activation)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)


        self.dec_fc = nn.Linear(self.nhid, ntoken)
        self.generator = nn.Softmax(dim=-1)
        self.word_emb = nn.Embedding(ntoken, self.nhid)
        self.init_weights()
        if config.word_embedding.freeze:
            self.word_emb.weight.requires_grad = False


    def init_weights(self):
        initrange = 0.1
        self.word_emb.weight.data.uniform_(-initrange, initrange)
        self.dec_fc.bias.data.zero_()
        self.dec_fc.weight.data.uniform_(-initrange, initrange)

    #complete encoder of transformer
    def encode(self, src, mixup_param=None):
        #CNN10 is our feature extractor
        src = self.feature_extractor(src, mixup_param)
        #linear audio data
        src = self.audio_linear(src)
        src = src * math.sqrt(self.nhid)
        #positional encoder
        src = self.pos_encoder(src)
        #transformer encoder
        src = self.transformer_encoder(src, None)
        return src

    #complete decoder of transformer
    def decode(self, mem, tgt, input_mask=None, target_mask=None, target_padding_mask=None):
        tgt = tgt.transpose(0, 1)
        tgt = self.word_emb(tgt) * math.sqrt(self.nhid)
        tgt = self.pos_encoder(tgt)
        output = self.transformer_decoder(tgt, mem,
                                          memory_mask=input_mask,
                                          tgt_mask=target_mask,
                                          tgt_key_padding_mask=target_padding_mask)
        output = self.dec_fc(output)
        return output

    #main forward pass
    def forward(self, src, tgt, input_mask=None, target_mask=None, target_padding_mask=None):
        #calling encoder
        mem = self.encode(src)
        #calling decoder
        output = self.decode(mem, tgt,
                             input_mask=input_mask,
                             target_mask=target_mask,
                             target_padding_mask=target_padding_mask)
        return output
        
        
def train():
    start_time = time.time()
    batch_losses = torch.zeros(len(training_data))
    model.train()
    for batch_idx, train_batch in tqdm(enumerate(training_data), total=len(training_data)): #tqdm = progress bar
        src, tgt, f_names, tgt_len, captions = train_batch
        src = src.to(device)
        tgt = tgt.to(device)
        #masking data
        tgt_pad_mask = set_tgt_padding_mask(tgt, tgt_len)
        optimizer.zero_grad()
        #prediction from model
        y_hat = model(src, tgt, target_padding_mask=tgt_pad_mask)
        #real target value
        tgt = tgt[:, 1:]
        y_hat = y_hat.transpose(0, 1)  # batch x words_len x ntokens
        y_hat = y_hat[:, :tgt.size()[1], :]
        #loss function
        loss = criterion(y_hat.contiguous().view(-1, y_hat.size()[-1]),
                         tgt.contiguous().view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.clip_grad)
        optimizer.step()
        batch_losses[batch_idx] = loss.cpu().item()
    
    end_time = time.time()
    elasped_time = end_time - start_time
    epoch_loss = batch_losses.mean()
    current_lr = [param_group['lr'] for param_group in optimizer.param_groups][0]
    main_logger.info('epoch: {}, train_loss: {:.4f}, time elapsed: {:.4f}, lr:{:02.2e}'.format(epoch, epoch_loss, elasped_time, current_lr))

def start_train():
    #collecting data from settings file
    config = get_config()
    logger.remove()
    logger.add(sys.stdout, format='{time: YYYY-MM-DD at HH:mm:ss} | {message}',
               level='INFO', filter=lambda record: record['extra']['indent'] == 1)
    main_logger = logger.bind(indent=1)
    printer = PrettyPrinter()
    device = "cpu"
    # device_name = (torch.device('cuda'), torch.cuda.get_device_name(torch.cuda.current_device()))
    dataset = config.data.type
    batch_size = config.data.batch_size
    num_workers = config.data.num_workers
    input_field_name = config.data.input_field_name

    # loading vocabulary list
    words_list_path = 'new_words_list.p'
    training_data = get_clotho_loader(split='development',
                                      input_field_name=input_field_name,
                                      load_into_memory=False,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=num_workers)
    words_list = load_picke_file(words_list_path)
    ntokens = len(words_list)
    sos_ind = words_list.index('<sos>')
    eos_ind = words_list.index('<eos>')

    #CNN10 modle
    pretrained_cnn = torch.load("Cnn10.pth",map_location=torch.device('cpu'))
    main_logger.info('Training setting:\n'
                     f'{printer.pformat(config)}')
    model = TransformerModel(config, words_list, pretrained_cnn)
    model.to(device)
    main_logger.info(f'Model:\n{model}\n')
    main_logger.info('Total number of parameters:'
                     f'{sum([i.numel() for i in model.parameters()])}')

    criterion = nn.CrossEntropyLoss()
    spiders = []

    main_logger.info('Training mode.')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.training.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.1)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)
    epochs = config.training.epochs
    ep = 1
    # warm up issue
    optimizer.zero_grad()
    optimizer.step()
    for epoch in range(ep, epochs + 1):
        scheduler_warmup.step(epoch)
        main_logger.info(f'Training epoch {epoch}...')
        train()
    main_logger.info('Training done.')