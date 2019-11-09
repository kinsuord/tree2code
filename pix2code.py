import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention
import numpy as np
from utils.transforms import Rescale, WordEmbedding, TreeToBfsSeq, SeqToTensor
from nltk.translate.bleu_score import corpus_bleu
from dataset import Pix2TreeDataset

# Data parameters
data_folder = '/media/ssd/caption data'  # folder with data files saved by create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files

# Model parameters
emb_dim = 14  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 120  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 32
workers = 1  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?
checkpoint = None  # path to checkpoint, None if none

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best):
    """
    Saves model checkpoint.
    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_' + data_name + '.pth'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def batch_collate(batch):
    out = dict()
    out['img'] = torch.utils.data.dataloader.default_collate(
                                            [x['img'] for x in batch])
    out['code'] = [x['tree'] for x in batch]

    max_len = max([x.size()[0] for x in out['code']])

    # magic zero padding to max_len
    out['len'] = [torch.IntTensor([x.size()[0]]) for x in out['code']]
    out['len'] = torch.stack(out['len']) 

    out['code'] = [torch.cat(
            [x.float(), torch.zeros(max_len-x.size()[0], x.size()[1])], dim=0)
            for x in out['code']]
    out['code'] = torch.stack(out['code'], dim=1)
    out['code'] = out['code'].permute(1,0,2)
    return out

def main():
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map


    word_map = np.load('word_dict.npy', allow_pickle=True).item()
    dataset = Pix2TreeDataset()
    train_data = Pix2TreeDataset(partition=range(int(len(dataset)*0.8)),
                                 tree_transform=transforms.Compose([
                                         WordEmbedding(word_map),
                                         TreeToBfsSeq(),
                                         SeqToTensor()]),
            img_transform=transforms.Compose([Rescale(224),
                                              transforms.ToTensor()]))
    valid_data = Pix2TreeDataset(
            partition=range(int(len(dataset)*0.8), len(dataset)),
            img_transform=transforms.Compose([Rescale(224),
                                              transforms.ToTensor()]))
    
    # Initialize / load checkpoint
    if checkpoint is None:
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.BCELoss()

    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                        pin_memory=True, shuffle=True,
                        collate_fn=batch_collate, num_workers=workers)
    
    val_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, 
                        pin_memory=True, shuffle=True,
                        collate_fn=batch_collate, num_workers=workers)


    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # # One epoch's validation
        # recent_bleu4 = validate(val_loader=val_loader,
        #                         encoder=encoder,
        #                         decoder=decoder,
        #                         criterion=criterion)

        # Check if there was an improvement
        # is_best = recent_bleu4 > best_bleu4
        # best_bleu4 = max(recent_bleu4, best_bleu4)
        # if not is_best:
        #     epochs_since_improvement += 1
        #     print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        # else:
        #     epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, None, None)

def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)

def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.
    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, batch in enumerate(train_loader):
        data_time.update(time.time() - start)
        # import pdb; pdb.set_trace()

        # Move to GPU, if available
        imgs = batch['img'].to(device)
        caps = batch['code'].to(device)
        caplens = batch['len'].to(device)

        # Forward prop.
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Calculate loss
        loss = criterion(scores.data.float(), targets.data.to('cpu'))
        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        scores = scores.data
        targets = targets.data
        # Keep track of metrics
        # top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        # top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses))


def validate(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    import pdb; pdb.set_trace()
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            # top5 = accuracy(scores, targets, 5)
            # top5accs.update(top5, sum(decode_lengths))
            # batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

    return bleu4


if __name__ == '__main__':
    main()


#from torch.utils.data import DataLoader
#from torchvision import transforms
#import torch
#import datetime
#import numpy as np
#import os
#import torch.backends.cudnn as cudnn
#
#from models import Encoder, DecoderWithAttention
#from dataset import Pix2TreeDataset
#from utils.transforms import Rescale, WordEmbedding, TreeToBfsSeq, SeqToTensor
#from utils.generator import Env
#
#def train(name, train_data, word_dict, checkpoint=None):
#    # Model parameters
#    emb_dim = 512  # dimension of word embeddings
#    attention_dim = 512  # dimension of attention linear layers
#    decoder_dim = 512  # dimension of decoder RNN
#    dropout = 0.5
#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
#    cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
#    
#    # Training parameters
#    start_epoch = 0
#    epochs = 120  # number of epochs to train for (if early stopping is not triggered)
#    epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
#    batch_size = 32
#    workers = 2  # for data-loading; right now, only 1 works with h5py
#    encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
#    decoder_lr = 4e-4  # learning rate for decoder
#    grad_clip = 5.  # clip gradients at an absolute value of
#    alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
#    best_bleu4 = 0.  # BLEU-4 score right now
#    print_freq = 100  # print training/validation stats every __ batches
#    fine_tune_encoder = False  # fine-tune encoder?
#    
#    dataloader = DataLoader(train_data, batch_size=batch_size, 
#                        pin_memory=True, shuffle=True,
#                        collate_fn=batch_collate, num_workers=workers)
#    
#    # Initialize / load checkpoint
#    if checkpoint is None:
#        decoder = DecoderWithAttention(attention_dim=attention_dim,
#                                       embed_dim=emb_dim,
#                                       decoder_dim=decoder_dim,
#                                       vocab_size=len(word_dict),
#                                       dropout=dropout)
#        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
#                                             lr=decoder_lr)
#        encoder = Encoder()
#        encoder.fine_tune(fine_tune_encoder)
#        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
#                                             lr=encoder_lr) if fine_tune_encoder else None
#
#    else:
#        checkpoint = torch.load(checkpoint)
#        start_epoch = checkpoint['epoch'] + 1
#        epochs_since_improvement = checkpoint['epochs_since_improvement']
#        best_bleu4 = checkpoint['bleu-4']
#        decoder = checkpoint['decoder']
#        decoder_optimizer = checkpoint['decoder_optimizer']
#        encoder = checkpoint['encoder']
#        encoder_optimizer = checkpoint['encoder_optimizer']
#        if fine_tune_encoder is True and encoder_optimizer is None:
#            encoder.fine_tune(fine_tune_encoder)
#            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
#                                                 lr=encoder_lr)
#
#    decoder = decoder.to(device)
#    encoder = encoder.to(device)
#    criterion = torch.nn.CrossEntropyLoss().to(device)
#    
#    # Epochs
#    for epoch in range(start_epoch, epochs):
#
#        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
#        if epochs_since_improvement == 20:
#            break
#        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
#            adjust_learning_rate(decoder_optimizer, 0.8)
#            if fine_tune_encoder:
#                adjust_learning_rate(encoder_optimizer, 0.8)
#
#        # One epoch's training
#        train(train_loader=train_loader,
#              encoder=encoder,
#              decoder=decoder,
#              criterion=criterion,
#              encoder_optimizer=encoder_optimizer,
#              decoder_optimizer=decoder_optimizer,
#              epoch=epoch)
#
#        # One epoch's validation
#        recent_bleu4 = validate(val_loader=val_loader,
#                                encoder=encoder,
#                                decoder=decoder,
#                                criterion=criterion)
#
#        # Check if there was an improvement
#        is_best = recent_bleu4 > best_bleu4
#        best_bleu4 = max(recent_bleu4, best_bleu4)
#        if not is_best:
#            epochs_since_improvement += 1
#            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
#        else:
#            epochs_since_improvement = 0
#
#        # Save checkpoint
#        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
#                        decoder_optimizer, recent_bleu4, is_best)
#
#def validate(val_loader, encoder, decoder, criterion):
#    """
#    Performs one epoch's validation.
#    :param val_loader: DataLoader for validation data.
#    :param encoder: encoder model
#    :param decoder: decoder model
#    :param criterion: loss layer
#    :return: BLEU-4 score
#    """
#    decoder.eval()  # eval mode (no dropout or batchnorm)
#    if encoder is not None:
#        encoder.eval()
#
#    batch_time = AverageMeter()
#    losses = AverageMeter()
#    top5accs = AverageMeter()
#
#    start = time.time()
#
#    references = list()  # references (true captions) for calculating BLEU-4 score
#    hypotheses = list()  # hypotheses (predictions)
#
#    # explicitly disable gradient calculation to avoid CUDA memory error
#    # solves the issue #57
#    with torch.no_grad():
#        # Batches
#        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):
#
#            # Move to device, if available
#            imgs = imgs.to(device)
#            caps = caps.to(device)
#            caplens = caplens.to(device)
#
#            # Forward prop.
#            if encoder is not None:
#                imgs = encoder(imgs)
#            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)
#
#            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
#            targets = caps_sorted[:, 1:]
#
#            # Remove timesteps that we didn't decode at, or are pads
#            # pack_padded_sequence is an easy trick to do this
#            scores_copy = scores.clone()
#            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
#            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)
#
#            # Calculate loss
#            loss = criterion(scores, targets)
#
#            # Add doubly stochastic attention regularization
#            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
#
#            # Keep track of metrics
#            losses.update(loss.item(), sum(decode_lengths))
#            top5 = accuracy(scores, targets, 5)
#            top5accs.update(top5, sum(decode_lengths))
#            batch_time.update(time.time() - start)
#
#            start = time.time()
#
#            if i % print_freq == 0:
#                print('Validation: [{0}/{1}]\t'
#                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
#                                                                                loss=losses, top5=top5accs))
#
#            # Store references (true captions), and hypothesis (prediction) for each image
#            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
#            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
#
#            # References
#            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
#            for j in range(allcaps.shape[0]):
#                img_caps = allcaps[j].tolist()
#                img_captions = list(
#                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
#                        img_caps))  # remove <start> and pads
#                references.append(img_captions)
#
#            # Hypotheses
#            _, preds = torch.max(scores_copy, dim=2)
#            preds = preds.tolist()
#            temp_preds = list()
#            for j, p in enumerate(preds):
#                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
#            preds = temp_preds
#            hypotheses.extend(preds)
#
#            assert len(references) == len(hypotheses)
#
#        # Calculate BLEU-4 scores
#        bleu4 = corpus_bleu(references, hypotheses)
#
#        print(
#            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
#                loss=losses,
#                top5=top5accs,
#                bleu=bleu4))
#
#    return bleu4
#
#if __name__ == '__main__':
#    
#    # load word dict
#    def count_word_dict(dataset):
#        word_count = {'root':0, 'end':0}
#        def count_tree(tree, word_count):
#            for child in tree.children:
#                count_tree(child, word_count)
#            if tree.value in word_count:
#                word_count[tree.value] += 1
#            else:
#                word_count[tree.value] = 1
#        
#        for i in range(len(dataset)):
#            count_tree(dataset[i]['tree'], word_count)
#        
#        word_dict = {}
#        i = 0
#        for key in word_count.keys():
#            a = np.zeros(len(word_count))
#            a[i] = 1.0
#            word_dict[key] = a
#            i += 1
#        return word_dict
#
#    dataset = Pix2TreeDataset()
#    if not os.path.exists('word_dict.npy'):
#        word_dict = count_word_dict(dataset)
#        np.save('word_dict.npy', word_dict) 
#    else:
#        word_dict = np.load('word_dict.npy', allow_pickle=True).item()
#
#    train_data = Pix2TreeDataset(partition=range(int(len(dataset)*0.8)),
#            tree_transform=transforms.Compose([WordEmbedding(word_dict),
#                                               TreeToBfsSeq(),
#                                               SeqToTensor()]),
#            img_transform=transforms.Compose([Rescale(224),
#                                              transforms.ToTensor()]))
#
#    train('seqWithAttention', train_data, word_dict)
##    tree2code_model = Pix2CodeModel(len(word_dict))
##    train('pixToCode', tree2code_model, train_data, epoch=30, batch_size=32,
##            lr = 10e-9)
#
#
#def batch_collate(batch):
#    out = dict()
#    out['img'] = torch.utils.data.dataloader.default_collate(
#                                            [x['img'] for x in batch])
#    out['code'] = [x['tree'] for x in batch]
#
#    max_len = max([x.size()[0] for x in out['code']])
#
#    # magic zero padding to max_len
#    out['code'] = [torch.cat(
#            [x.float(), torch.zeros(max_len-x.size()[0], x.size()[1])], dim=0)
#            for x in out['code']]
#    out['code'] = torch.stack(out['code'], dim=1)
#    return out
#
##def train(save_name, model, train_data, pretrain_epoch=0, epoch=2, lr=1e-5, 
##          batch_size=1, num_worker=1, 
##          loss_freq=10, save_freq=10):
##    
##    # Model parameters
##    emb_dim = 512  # dimension of word embeddings
##    attention_dim = 512  # dimension of attention linear layers
##    decoder_dim = 512  # dimension of decoder RNN
##    dropout = 0.5
##    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
##    cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
##
##    dataloader = DataLoader(train_data, batch_size=batch_size, 
##                            pin_memory=True, shuffle=True,
##                            collate_fn=batch_collate, num_workers=num_worker)
##    
##    # init model
##    if pretrain_epoch == 0:
##        decoder = DecoderWithAttention(attention_dim=attention_dim,
##                                       embed_dim=emb_dim,
##                                       decoder_dim=decoder_dim,
##                                       vocab_size=len(word_map),
##                                       dropout=dropout)
##        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
##                                             lr=decoder_lr)
##        encoder = Encoder()
##        encoder.fine_tune(fine_tune_encoder)
##        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
##                                             lr=encoder_lr) if fine_tune_encoder else None
##    else:
##        checkpoint = torch.load('{}_{}'.format(save_name, pretrain_epoch))
##        start_epoch = checkpoint['epoch'] + 1
##        epochs_since_improvement = checkpoint['epochs_since_improvement']
##        best_bleu4 = checkpoint['bleu-4']
##        decoder = checkpoint['decoder']
##        decoder_optimizer = checkpoint['decoder_optimizer']
##        encoder = checkpoint['encoder']
##        encoder_optimizer = checkpoint['encoder_optimizer']
##        if fine_tune_encoder is True and encoder_optimizer is None:
##            encoder.fine_tune(fine_tune_encoder)
##            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
##                                                 lr=encoder_lr)
##
##    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
##    criterion = nn.CrossEntropyLoss()
##    pretrain_e = 0
##    if pretrain == None:
##        # initialize
##        def weights_init_uniform_rule(m):
##            classname = m.__class__.__name__
##            # for every Linear layer in a model..
##            if classname.find('Linear') != -1:
##                # get the number of the inputs
##                n = m.in_features
##                y = 1.0/np.sqrt(n)
##                m.weight.data.uniform_(-y, y)
##                m.bias.data.fill_(0)
##    
##        model.apply(weights_init_uniform_rule)
##    else:
##        checkpoint = torch.load(pretrain)
##        model.load_state_dict(checkpoint['model_state_dict'])
##        pretrain_e = checkpoint['epoch']
##        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
##        for state in optimizer.state.values():
##            for k, v in state.items():
##                if isinstance(v, torch.Tensor):
##                    state[k] = v.to(device)
##        
##    model.to(device)
##    model.train()
##    for e in range(pretrain_e, pretrain_e + epoch):
##        for batch_i, batch_sample in enumerate(dataloader):
##            import pdb; pdb.set_trace()
###            img = batch_sample['img']
###            code = batch_sample['code']
###            code = code.to(device)
###            img = img.to(device)
###            pred = model(img, code)
###
###            mask = []
###            for i in range(code.size(0)):
###                mask1 = []
###                for j in range(code.size(1)):
###                    if sum(code[i][j]) == 0.0:
###                        mask1.append(torch.zeros(code.size(2)).to(device))
###                    else:
###                        mask1.append(torch.ones(code.size(2)).to(device))
###                mask1 = torch.stack(mask1)
###                mask.append(mask1)
###
###            mask = torch.stack(mask, dim=0)
###            # import pdb; pdb.set_trace()
###                        
###            pred = pred * mask
###            # code[1:] = torch.where(mask[1:] == 1, 
###            #     code[1:], pred[:-1])
###
###            torch.autograd.set_detect_anomaly(True)
###            # answer = torch.clone(code[1:]).detach()
###
###            optimizer.zero_grad()
###            loss = criterion(pred[:-1], code[1:])
###            loss.backward()
###            optimizer.step()
###
###            if batch_i%loss_freq == 0:
###                print('epoch:{} batch_1:{} loss:{}'.format(
###                        e, batch_i, loss))
###
###            # if batch_i%save_freq == 0:
###            #     checkpoint_path = os.path.join(
###            #         'checkpoint', '{}_{}_{}.pth'.format(save_name, e, batch_i))
###            #     torch.save({
###            #         'epoch': epoch,
###            #         'model_state_dict': model.state_dict(),
###            #         'optimizer_state_dict': optimizer.state_dict(),
###            #         'loss':  loss
###            #     }, checkpoint_path)
###            #     print(datetime.datetime.now(), 'save model to {}'.
###            #                                          format(checkpoint_path))
###        
###        if batch_i%save_freq == 0:    
###            checkpoint_path = os.path.join(
###                    'checkpoint', '{}_{}.pth'.format(save_name, e))
###            torch.save({
###                'epoch': e,
###                'model_state_dict': model.state_dict(),
###                'optimizer_state_dict': optimizer.state_dict(),
###                'loss': loss
###            }, checkpoint_path)
###            print(datetime.datetime.now(), 'save model to {}'.
###                                                format(checkpoint_path))
##    return model
##
##
