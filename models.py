# -*- coding: utf-8 -*-

import torch
import torch.nn as nn 
import torchvision.models as models
from utils.tree import Tree

class ChildSumTreeLSTM(nn.Module):
    '''
    input: Tree(in_dim)
    output: state(mem_dim)
    '''
    def __init__(self, in_dim, mem_dim):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)

    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)
        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

        f = torch.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, torch.tanh(c))
        return c, h

    def forward(self, tree):
        for idx in range(tree.num_children):
            self.forward(tree.children[idx])

        inputs = tree.value
        if tree.num_children == 0:
            child_c = inputs.detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
            child_h = inputs.detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
        else:
            child_c, child_h = zip(* map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(inputs, child_c, child_h)
        return tree.state

class ChildSumLSTMCell(nn.Module):
    '''
    input: inputs(word_dim), child_c, child_h
    output: c, h
    '''
    def __init__(self, in_dim, mem_dim):
        super(ChildSumLSTMCell, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)

    def forward(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)
        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

        f = torch.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, torch.tanh(c))
        return c, h


class Pix2TreeReverse(nn.Module):
    '''
    input: img(224, 224, 3), Tree(word_dim), parent of next_word (Tree(word dim))
    input: img(224, 224, 3), List(Tree): path to next_node parent
    output: Seqence( (root) form  node -> node -> next_word)
    '''
    def __init__(self, word_dim, mem_dim):
        super(Pix2TreeReverse, self).__init__()
        self.word_dim = word_dim
        self.mem_dim = mem_dim
        self.img_dim = 1000

        self.cnn = models.vgg16(pretrained=True) # 1000
        self.img_to_c = nn.Linear(self.img_dim, self.mem_dim)
        self.tree_lstm = ChildSumLSTMCell(self.word_dim, self.mem_dim)

        self.h_to_word = nn.Sequential(            
            nn.Linear(self.mem_dim, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, self.word_dim),
            nn.Softmax()
        )

    def forward(self, img, path):
        
        output = []
        for i in range(len(path)):
            # if path[i] = root
            if path[i].parent == None:
                img_features = self.cnn(img)
                child_c = self.img_to_c(img_features)
                child_h = path[0].value.detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
                dummy = Tree('')
                dummy.state = child_c, child_h
                feed_list = [dummy]
            else:
                feed_list = [path[i].parent]
                idx = path[i].parent.children.index(path[i])
                for j in range(idx):
                    child = path[i].parent.children[j]
                    child_c = child.value.detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
                    child_h = child.value.detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
                    child.state = self.tree_lstm(child.value, child_c, child_h)
                    feed_list.append(child)
            
            # if this is the end
            if i == len(path)-1:
                idx = len(path[i].children)
            else:
                idx = path[i].children.index(path[i+1])
            
            for j in range(idx):
                child = path[i].children[j]
                child_c = child.value.detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
                child_h = child.value.detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
                child.state = self.tree_lstm(child.value, child_c, child_h)
                feed_list.append(child)

            child_c, child_h = zip(* map(lambda x: x.state, feed_list))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)
            
            path[i].state = self.tree_lstm(path[i].value, child_c, child_h)
            output.append(self.h_to_word(path[i].state[1]))

        return torch.cat(output, dim=0)
        


class ShowAndTellTree(nn.Module):
    '''
    input: img(224, 224, 3), Tree(word_dim)
    output: next_word(word_dim)
    '''
    def __init__(self, word_dim):
        super(ShowAndTellTree, self).__init__()
        self.word_dim = word_dim
        
        self.tree_lstm = ChildSumTreeLSTM(word_dim, 1024)
        self.cnn = models.vgg16(pretrained=True) # 1000
        self.fc = nn.Sequential(
            nn.Linear(2024, 2024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2024, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, word_dim),
            nn.Softmax()
        )
        
    def forward(self, img, tree):
        img_features = self.cnn(img)
        tree_state, _ = self.tree_lstm(tree)
        return self.fc(torch.cat((img_features, tree_state), dim=1))

class BatchModel(nn.Module):
    '''
    input: img(batch_size, 224, 224, 3), List(Tree())(length: batch_size)
    output: pred(batch_size, word_dim)
    '''
    def __init__(self, word_dim):
        super(BatchModel, self).__init__()
        self.word_dim = word_dim
        
        self.tree_lstm = ChildSumTreeLSTM(word_dim, 512)
        self.cnn = models.vgg11_bn(pretrained=True) # 1000
        self.fc = nn.Sequential(
            nn.Linear(1512, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, word_dim),
            nn.Softmax()
        )
        
    def forward(self, img, tree):
        img_features = self.cnn(img)
        tree_features = []
        for i in range(len(tree)):
            tree_features.append(self.tree_lstm(tree[i])[0])
        tree_features = torch.cat(tree_features, dim=0)
        return self.fc(torch.cat((img_features, tree_features), dim=1))
    
class Pix2TreeKai(nn.Module):
    '''
    input: img(batch_size, 224, 224, 3), 
            List(Tree())(length: batch_size)
            Parent(length: batch_size, word_dim)
    output: pred(batch_size, word_dim)
    '''
    def __init__(self, word_dim):
        super(Pix2TreeKai, self).__init__()
        self.word_dim = word_dim
        
        self.tree_lstm = ChildSumTreeLSTM(word_dim, 512)
        self.cnn = models.vgg11_bn(pretrained=True) # 1000
        self.fc = nn.Sequential(
            nn.Linear(1512 + word_dim, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, word_dim),
            nn.Softmax()
        )
        
    def forward(self, img, tree, parent):
        img_features = self.cnn(img)
        tree_features = []
        for i in range(len(tree)):
            tree_features.append(self.tree_lstm(tree[i])[0])
        tree_features = torch.cat(tree_features, dim=0)
        # print("img: {} parent: {}".format(img_features.size(), parent.size()))
        return self.fc(torch.cat((img_features, tree_features, parent), dim=1))

    
class Pix2CodeModel(nn.Module):
    '''
        input: img(batch_size, 224, 224, 3), code(seq_len, batch, word_dim)
        output: pred(batch_size, word_dim)
    '''
    def __init__(self, word_dim):
        super(Pix2CodeModel, self).__init__()
        self.img_conv = nn.Sequential(
            nn.Conv2d(3,32,3),
            nn.ReLU(),
            nn.Conv2d(32,32,3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(32,64,3),
            nn.ReLU(),
            nn.Conv2d(64,64,3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(64,128,3),
            nn.ReLU(),
            nn.Conv2d(128,128,3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

        self.img_dense = nn.Sequential(
            nn.Linear(73728, 1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.Dropout(0.3)
        )

        self.code_encode = torch.nn.LSTM(14, 128, 2)
        self.code_decode = torch.nn.LSTM(1152, 512, 2)
        self.dense_decode = nn.Sequential(
            nn.Linear(512, word_dim),
            nn.Softmax()
        )

    def forward(self, img, code):
        img = self.img_conv(img)
        img_encode = self.img_dense(img.view(img.size(0), -1))
        img_encode = torch.stack([img_encode]*code.size(0),dim=0)
        code_encode, _ = self.code_encode(code)
        decoder = torch.cat((img_encode, code_encode), dim=2)
        # import pdb; pdb.set_trace()
        pred, _ = self.code_decode(decoder)
        return self.dense_decode(pred)
    
  
class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Sequential(
            nn.Linear(decoder_dim, vocab_size),
            nn.Softmax()  # linear layer to find scores over vocabulary
        )
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc[0].bias.data.fill_(0)
        self.fc[0].weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.
        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # # Embedding
        # embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()
        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([encoded_captions[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind