import torch
from torch import nn, optim
from Data_loader import Data_for_train
from Data_loader import freinds_parsing
import random
from losses import KLD, similarity_loss
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch.nn.functional as F
import argparse

from model_creation import vae_auto_encoder

def prepare_sequence_data(charecters=None, sequence_length=15, dict_data=None):
    _train = {}
    i = 0
    trainning = []
    label_data = []
    for ind, name in enumerate(charecters):
        dict = dict_data[name]
        # (dict)
        for vals in dict.values():
            val_temp = np.zeros(sequence_length + 1, dtype=np.int)
            val_temp[0] = label_charecter[ind]
            if len(vals) >= sequence_length:
                for ii in range(1, np.min([20, len(vals) - sequence_length]), 5):
                    label_zero = np.zeros((len(charecters)), dtype=np.int)
                    _train[i] = {}
                    label_zero[ind] = 1
                    _train[i]['label'] = label_zero.copy()
                    _train[i]['values'] = vals[ii:sequence_length + ii]
                    val_temp[1:sequence_length + 1] = vals[ii:sequence_length + ii]
                    if len(trainning) == 0:
                        trainning = np.expand_dims(val_temp.copy(), axis=0)
                        label_data = np.expand_dims(label_zero, axis=0)
                    else:
                        trainning = np.concatenate([trainning, np.expand_dims(val_temp.copy(), axis=0)], axis=0)
                        label_data = np.concatenate([label_data, np.expand_dims(label_zero, axis=0)], axis=0)
                    i = i + 1
    return trainning, label_data

def url_chooser(name='freinds', debug_mode=False):
    if name == 'freinds':
        url_list = freinds_parsing()
    if name == 'lion_king':
        url_list = [r'https://transcripts.fandom.com/wiki/The_Lion_King',
                    r'https://transcripts.fandom.com/wiki/The_Lion_King_II:_Simba%27s_Pride',
                    r'https://transcripts.fandom.com/wiki/Never_Everglades',
                    r'http://www.lionking.org/scripts/TLK1.5-Script.html#specialpower']
    if name == 'games_of_thorne':
        url_list = [r'https://genius.com/Game-of-thrones-the-pointy-end-annotated',
                    r'https://genius.com/Game-of-thrones-the-pointy-end-annotated',
                    r'https://genius.com/Game-of-thrones-the-kingsroad-annotated', \
                    r'https://genius.com/Game-of-thrones-cripples-bastards-and-broken-things-annotated',
                    r'https://genius.com/Game-of-thrones-the-wolf-and-the-lion-annotated', \
                    r'https://genius.com/Game-of-thrones-a-golden-crown-annotated',
                    r'https://genius.com/Game-of-thrones-you-win-or-you-die-annotated',
                    r'https://genius.com/Game-of-thrones-fire-and-blood-annotated',
                    r'https://genius.com/Game-of-thrones-the-north-remembers-annotated', \
                    r'https://genius.com/Game-of-thrones-the-night-lands-annotated',
                    r'https://genius.com/Game-of-thrones-what-is-dead-may-never-die-annotated',
                    r'https://genius.com/Game-of-thrones-garden-of-bones-annotated', \
                    r'https://genius.com/Game-of-thrones-the-ghost-of-harrenhal-annotated',
                    r'https://genius.com/Game-of-thrones-the-old-gods-and-the-new-annotated',
                    r'https://genius.com/Game-of-thrones-a-man-without-honor-annotated', \
                    r'https://genius.com/Game-of-thrones-the-prince-of-winterfell-annotated',
                    r'https://genius.com/Game-of-thrones-blackwater-annotated',
                    r'https://genius.com/Game-of-thrones-valar-morghulis-annotated', \
                    r'https://genius.com/Game-of-thrones-valar-dohaeris-annotated']
    if name == 'Harry_potter':
        url_list = [r'https://www.hogwartsishere.com/library/book/12647/chapter/1/',
                    r'https://www.hogwartsishere.com/library/book/12647/chapter/2/' \
                    r'https://www.hogwartsishere.com/library/book/12647/chapter/3/']
    if debug_mode:  # subset of data just to debug code
        url_list = url_list[0:2]
    return url_list

def generation_func(data_train, decoder, encoder, charecter='joey', gen_length=11, sentence=None):
    decoder_context = torch.zeros(1, encoder.hidden_size * (1 + encoder.bidirectional)).cuda()

    encoded_list = []
    output_text = []
    charecter_ind = data_train.word2index[charecter + '1']
    batch_size = 1
    hidden = encoder.init_hidden(batch_size=batch_size)

    decoder_hidden = decoder.init_hidden(batch_size=batch_size)

    encoded_list.append(charecter_ind)
    for i in range(len(sentence)):
        encoded_list.append(data_train.word2index[sentence[i]])
    encoder_input = torch.LongTensor(encoded_list).cuda()

    for k in range(len(sentence) + 1):
        decoder_input = encoder_input[k]
        encoder_output, hidden_encoder = encoder.forward(encoder_input.view(batch_size, len(encoded_list)),
                                                         last_hidden=hidden)

        output, decoder_hidden = decoder(decoder_input.view(batch_size, 1)
                                         , decoder_hidden, encoder_output)
    topv, topi = output.topk(k=3)
    # probs = torch.softmax(topv,2)

    choices = topi.tolist()
    ni = np.random.choice(choices[0])
    decoder_input = torch.LongTensor([ni]).cuda()
    encoded_list.append(ni)
    encoder_input = torch.LongTensor(encoded_list).cuda()
    output_text.append(data_train.index2word[ni])

    for i in range(gen_length):
        encoder_output, hidden_encoder = encoder.forward(encoder_input.view(batch_size, len(encoded_list)),
                                                         last_hidden=hidden)

        output, decoder_hidden = decoder(decoder_input.view(batch_size, 1),
                                         decoder_hidden, encoder_output)

        topv, topi = output.topk(k=3)
        # probs = torch.softmax(topv,2)

        ni = topi[0][0].item()
        choices = topi.tolist()
        ni = np.random.choice(choices[0])
        decoder_input = torch.LongTensor([ni]).cuda()
        encoded_list.append(ni)
        encoder_input = torch.LongTensor(encoded_list).cuda()
        output_text.append(data_train.index2word[ni])

    return output_text

    # target_tensor2 = torch.zeros(batch_size,data_train.n_words).cuda()
    # for lk in range(batch_size):
    #    target_tensor2[i,target_tensor[lk,ii]] =1
    # sentence_loss+= torch.sum(torch.abs(output.squeeze()-target_tensor2))
    output_softmax = torch.log_softmax((output.view(-1, batch_size, data_train.n_words)), 2).squeeze()

def generation_func_combination(data_train, decoder, encoder, charecters=['joey', 'rachel'], gen_length=11,
                                sentence=None):
    encoded_list1 = []
    encoded_list2 = []

    output_text = []
    charecter_ind1 = data_train.word2index[charecters[0] + '1']
    charecter_ind2 = data_train.word2index[charecters[1] + '1']

    batch_size = 1
    hidden = encoder.init_hidden(batch_size=batch_size)

    decoder_hidden = decoder.init_hidden(batch_size=batch_size)

    encoded_list1.append(charecter_ind1)
    encoded_list2.append(charecter_ind2)

    for i in range(len(sentence)):
        encoded_list1.append(data_train.word2index[sentence[i]])
        encoded_list2.append(data_train.word2index[sentence[i]])

    encoder_input1 = torch.LongTensor(encoded_list1).cuda()
    encoder_input2 = torch.LongTensor(encoded_list2).cuda()

    for k in range(len(sentence)):
        encoder_output1, hidden_encoder1, mu, logvar = encoder.forward(
            encoder_input1.view(batch_size, len(encoded_list1)),
            last_hidden=hidden)
        encoder_output2, hidden_encoder2, mu, logvar = encoder.forward(
            encoder_input2.view(batch_size, len(encoded_list2)),
            last_hidden=hidden)
        decoder_input = encoder_input1[k + 1]  # its the same start word

        encoder_output = (encoder_output1[0:] + encoder_output2[
                                                0:]) / 2  # in order to skip the identity of  a speaker in the decoder.

        output, decoder_hidden = decoder(decoder_input.view(batch_size, 1),
                                         decoder_hidden, encoder_output)
    topv, topi = output.topk(k=3)
    # probs = torch.softmax(topv,2)

    choices = topi.tolist()
    ni = np.random.choice(choices[0])
    decoder_input = torch.LongTensor([ni]).cuda()
    encoded_list1.append(ni)
    encoded_list2.append(ni)

    encoder_input1 = torch.LongTensor(encoded_list1).cuda()
    encoder_input2 = torch.LongTensor(encoded_list2).cuda()

    output_text.append(data_train.index2word[ni])

    for i in range(gen_length):
        encoder_output1, hidden_encoder1, mu, logvar = encoder.forward(
            encoder_input1.view(batch_size, len(encoded_list1)), last_hidden=hidden)
        encoder_output2, hidden_encoder2, mu, logvar = encoder.forward(
            encoder_input2.view(batch_size, len(encoded_list2)), last_hidden=hidden)

        encoder_output = (encoder_output1[0:] + encoder_output2[
                                                0:]) / 2  # in order to skip the identity of  a speaker in the decoder.

        output, decoder_hidden = decoder(decoder_input.view(batch_size, 1)
                                         , decoder_hidden, encoder_output)

        topv, topi = output.topk(k=3)
        # probs = torch.softmax(topv,2)

        ni = topi[0][0].item()
        choices = topi.tolist()
        flage = 1
        while (flage):
            ni = np.random.choice(choices[0])
            decoder_input = torch.LongTensor([ni]).cuda()
            encoded_list1.append(ni)
            encoded_list2.append(ni)

            encoder_input1 = torch.LongTensor(encoded_list1).cuda()
            encoder_input2 = torch.LongTensor(encoded_list2).cuda()
            word_add = data_train.index2word[ni]
            if word_add != output_text[-1]:
                flage = 0
                output_text.append(data_train.index2word[ni])

    return output_text

    # target_tensor2 = torch.zeros(batch_size,data_train.n_words).cuda()
    # for lk in range(batch_size):
    #    target_tensor2[i,target_tensor[lk,ii]] =1
    # sentence_loss+= torch.sum(torch.abs(output.squeeze()-target_tensor2))
    output_softmax = torch.log_softmax((output.view(-1, batch_size, data_train.n_words)), 2).squeeze()
def generation_func_combination_random_sample(data_train, decoder , charecters=['joey', 'rachel'], gen_length=11,opt=None):
    encoded_list1 = []
    encoded_list2 = []

    output_text = []
    charecter_ind1 = data_train.word2index[charecters[0] + '1']
    charecter_ind2 = data_train.word2index[charecters[1] + '1']

    batch_size = 1

    decoder_hidden = decoder.init_hidden(batch_size=batch_size)

    encoded_list1.append(charecter_ind1)
    encoded_list2.append(charecter_ind2)
    decoder_input=torch.LongTensor(encoded_list1).cuda()
    encoder_output = torch.FloatTensor(np.random.random((1, 3, 2*opt.hidden_size))).cuda()

    for i in range(gen_length):

        output, decoder_hidden = decoder(decoder_input.view(batch_size, 1)
                                         , decoder_hidden, encoder_output)

        topv, topi = output.topk(k=3)
        choices = topi.tolist()
        ni = np.random.choice(choices[0])
        decoder_input = torch.LongTensor([ni]).cuda()
        encoded_list1.append(ni)
        encoded_list2.append(ni)

        output_text.append(data_train.index2word[ni])

    return output_text

    # target_tensor2 = torch.zeros(batch_size,data_train.n_words).cuda()
    # for lk in range(batch_size):
    #    target_tensor2[i,target_tensor[lk,ii]] =1
    # sentence_loss+= torch.sum(torch.abs(output.squeeze()-target_tensor2))
    output_softmax = torch.log_softmax((output.view(-1, batch_size, data_train.n_words)), 2).squeeze()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #model:
    parser.add_argument('--n_vocab', type=int, default=None)
    parser.add_argument('--n_hidden', type=int, default=20)
    parser.add_argument('--n_layers_E', type=int, default=2)
    parser.add_argument('--n_layers_D', type=int, default=2)

    #Dataset:
    parser.add_argument('--url', type=str, default='freinds')
    parser.add_argument('--charecters', type=list, default=['rachel', 'monica', 'joey', 'chandler', 'pheuby', 'ross'])
    parser.add_argument('--load_data', type=bool, default=True,help='After preprocessing the data a pickle is \\'
                                                                        'created in order to do it once')
    parser.add_argument('--embedding_size', type=int, default=50)
    parser.add_argument('--contain_non_embbedding', type=bool, default=True,help='in creation of the data to ignore non\\'
                                                                                 'vocabulary words')
    parser.add_argument('--path_to_pretrained_embedding', type=str, default='.\\embedding\\glove_vectors_50d.npy')
    parser.add_argument('--word2idx_path_embedding', type=str, default='.\\embedding\\wordsidx.txt')



    parser.add_argument('--sentence_L', type=int, default=20,help= 'The length of the sentence to process')
    parser.add_argument('--to_train', type=bool, default=True,help='if false only generation')
    parser.add_argument('--generation_type', type=str, default='normal',help='to train == False')
    parser.add_argument('--gen_length', type=int, default='20',help='to train == False , sentence length of generation')

    #Optimizer: (using cyclic LR)  \\
    parser.add_argument('--base_lr', type=float, default=8e-3)
    parser.add_argument('--max_lr', type=float, default=1e-2)

    # Trainning loop:
    parser.add_argument('--epochs', type=int, default=4000)
    parser.add_argument('--batch_size', type=int, default=1280)
    parser.add_argument('--tensorboard_dir', type=str, default='.\\tf\\anther_run\\')
    parser.add_argument('--load_pretrained', type=bool, default=True)
    parser.add_argument('--path_encoder', type=str, default='.\\saved_models\\encoder_3000')
    parser.add_argument('--path_decoder', type=str, default='.\\saved_models\\decoder_3000')
    parser.add_argument('--path_save_dir', type=str, default='.\\saved_models\\')



    parser.add_argument('--gpu_device', type=int, default=1)
    parser.add_argument('--debug_mode', type=bool, default=False,help='run only over a small part of the dataset')


    opt = parser.parse_args()

    ### Data creation:#######################################

    url_list = url_chooser(opt.url, debug_mode=opt.debug_mode)
    charecters = opt.charecters

    ## preparation of the data might take time. so it saves it. ( if it the first time running so load_data_set = False)
    load_data_set = opt.load_data
    if load_data_set == False:
        data_train = Data_for_train(url_list=url_list, charecters=charecters, contain_non_embbedding=opt.contain_non_embbedding,
                                    embedding_size=opt.embedding_size,embedding_path=opt.path_to_pretrained_embedding,word2idx_path=opt.word2idx_path_embedding)
        data_train.create_trainning_Data()
        # adding an embedding space which specify which charecter is speaking:
        np.save('trainning_data.npy', data_train)
    else:
        data_train = np.load('trainning_data.npy', allow_pickle='TRUE').item()

    dict_data = data_train.dict
    label_charecter = []
    # Adding to the vocabulary starting sentence charecter specific token:
    for char in opt.charecters:
        data_train.index_word(char + '1')
        label_charecter.append(data_train.word2index[char + '1'])
    dict_data = data_train.dict

    sequence_L = opt.sentence_L
    trainning, label_data = prepare_sequence_data(charecters=opt.charecters, sequence_length=sequence_L,
                                                  dict_data=dict_data)
    #########################################################################################################


    # Model parameters:

    encoder , decoder = vae_auto_encoder(opt, data_train)

    load = opt.load_pretrained
    path_encoder_model = opt.path_encoder
    path_decoder_model = opt.path_decoder  
    
    
    optimizer_decoder = optim.Adam(decoder.parameters(), lr=opt.base_lr)
    optimizer_encoder =  optim.Adam(encoder.parameters(), lr=opt.base_lr)

    if load == True:
        decoder_load = torch.load(path_decoder_model)
        encoder_load = torch.load(path_encoder_model)
        decoder.load_state_dict(decoder_load['model_state_dict'])
        encoder.load_state_dict(encoder_load['model_state_dict'])
        optimizer_decoder.load_state_dict(decoder_load['optimizer_state_dict'])
        optimizer_encoder.load_state_dict(encoder_load['optimizer_state_dict'])

    ##  splliting data for validation and trainning:
    
    rand_keys = np.random.permutation(len(trainning) - 1)
    train_num = int(0.95 * len(trainning))

    trainning_train = (trainning[rand_keys[0:train_num], :])
    label_train = label_data[rand_keys[0:train_num], :]

    #validation = (trainning[rand_keys[train_num:], :])
    #label_validation = label_data[rand_keys[train_num:], :]

    batch_size = opt.batch_size
    n_epochs = opt.epochs

    hidden = encoder.init_hidden(batch_size=batch_size)
    
    # Loss definning:
    loss = nn.NLLLoss()
    scheduler_decoder = CyclicLR(optimizer_decoder, base_lr=opt.base_lr, max_lr=opt.max_lr, step_size_up=1000, base_momentum=0.99,
                                 cycle_momentum=False)
    scheduler_encoder = CyclicLR(optimizer_encoder, base_lr=opt.base_lr, max_lr=opt.max_lr, step_size_up=1000, base_momentum=0.99,
                                 cycle_momentum=False)

    writer = SummaryWriter(opt.tensorboard_dir)

    weight = torch.FloatTensor([100, 20]).cuda()
    if opt.to_train == True:
        # Trainning loop :
        for e in range(1, n_epochs + 1):
            tot_loss = 0
            Total_batches = np.int(train_num / batch_size - 1)
            for i in range(Total_batches):
                decoder_hidden = decoder.init_hidden(batch_size=batch_size)
                sentence_loss = 0
                sequence_tensor = torch.LongTensor(trainning_train[i * batch_size: i * batch_size + batch_size, :]).cuda()
                target_tensor = torch.LongTensor(trainning_train[i * batch_size: i * batch_size + batch_size, 1:]).cuda()

                encoder_output, hidden_encoder, mu, logvar = encoder.forward(sequence_tensor, last_hidden=hidden)
                decoder_hidden = decoder.init_hidden(batch_size=batch_size)
                decoder_context = torch.zeros(batch_size, encoder.hidden_size * (1 + encoder.bidirectional)).cuda()
                for ii in range(sequence_L - 1):
                    decoder_input2 = torch.LongTensor(trainning_train[i * batch_size: i * batch_size + batch_size, ii]).cuda()
                    output, decoder_hidden = decoder(decoder_input2.view(batch_size, 1), decoder_hidden,
                                                          encoder_output)
                    output_softmax = torch.log_softmax((output.view(-1, batch_size, data_train.n_words)), 2).squeeze()
                    argmax = torch.argmax(output_softmax, 1)

                    targets = target_tensor[:, 1 + ii]

                    sentence_loss += loss(output_softmax, targets)*weight[0]+ KLD(mu,logvar) *weight[1]

                # updating weights
                # averaging total loss
                sentence_loss.backward()

                _ = torch.nn.utils.clip_grad_norm_(
                    decoder.parameters(), 5)
                _ = torch.nn.utils.clip_grad_norm_(
                    encoder.parameters(), 5)
                optimizer_decoder.step()
                optimizer_encoder.step()
                scheduler_decoder.step()
                scheduler_encoder.step()
                hidden[0].detach()
                hidden[1].detach()
                # sentence_loss.detach()
                optimizer_decoder.zero_grad()
                optimizer_encoder.zero_grad()
                tot_loss += sentence_loss.cpu().detach().numpy() / np.int(train_num / batch_size - 1)

                hidden = encoder.init_hidden(batch_size=batch_size)
                # hidden_encoder  = encoder.init_hidden(batch_size=batch_size)

            print('epoch:  ' + str(e))
            print('loss train')
            print(tot_loss)
            writer.add_scalar('trainning', global_step=e, scalar_value=tot_loss)
            writer.add_scalar('lr', global_step=e, scalar_value=scheduler_decoder.get_last_lr()[0])
            if np.mod(e, 100) == 0:
                torch.save({
                    'epoch': e,
                    'model_state_dict': decoder.state_dict(),
                    'optimizer_state_dict': optimizer_decoder.state_dict(),
                }, opt.path_save_dir+'decoder_' + str(e))
                # encoder model
                torch.save({
                    'epoch': e,
                    'model_state_dict': encoder.state_dict(),
                    'optimizer_state_dict': optimizer_encoder.state_dict(),
                }, opt.path_save_dir+'encoder_' + str(e))

    else: # For generation:
        print('Charecters list: '+str(opt.charecters) )
        charecters_choose = input("choose charecter anything")
        charecters_list = charecters_choose.split(sep=' ')
        if len(charecters_list) == 1:
            c1 = charecters_list[0]
            c2 = charecters_list[0]
        else:
            c1 = charecters_list[0]
            c2 = charecters_list[1]

        if opt.generation_type == 'random_sample':
            str_check=''
            while(str_check != 'no'):
                str_check = input("generate?")
                out_Str =generation_func_combination_random_sample(data_train, decoder , charecters=[c1, c2], gen_length=opt.gen_length,opt=opt)
                out = ''
                for i in range(len(out_Str)):
                    out += ' ' + out_Str[i]
                print('Generation:  ')
                print(out)
        else:
            str_check=''
            while(str_check != 'exit'):
                str_check = input("Type anything")
                ## Generation process example::
                start_string = str_check.split(sep=' ')
                input_string=[]
                for i in range(len(start_string)):
                    if start_string[i] in data_train.word2index:
                        input_string.append(start_string[i])
                    else:
                        print('Not in vocabulary: '+str(start_string[i])+' , choose anther word:')
                        str_check = 'asfd'
                        while(str_check not in data_train.word2index):
                            str_check = input("replace word:")
                        input_string.append(str_check)
                out = ''
                try:
                    out_Str = generation_func_combination(data_train, decoder=decoder, encoder=encoder, charecters=[c1, c2] \
                                                          , gen_length=20, sentence=input_string)
                except:
                    print('wrong input try agaiin')
                for i in range(len(input_string)):
                    out += ' ' + input_string[i]
                for i in range(len(out_Str)):
                    out += ' ' + out_Str[i]
                print('Generation:  ')
                print(out)