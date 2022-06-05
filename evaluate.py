import torch
import torch.nn as nn
from torchtext.data import Field, TabularDataset
import spacy
from utils import bleu,  get_exact_match
import random

# tokenizer
tokenize_code = lambda x: x.split()
spacy_eng = spacy.load("en_core_web_sm")
def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

# setup the Dataset
nl = Field(tokenize=tokenize_eng, lower=False, init_token="<sos>", eos_token="<eos>")
code = Field(tokenize=tokenize_code, lower=False, init_token="<sos>", eos_token="<eos>")
fields = {'nl': ('nl', nl), 'code': ('code', code)}
train_data, val_data = TabularDataset.splits(
                    path='./dataset',
                    train='train.json',
                    validation='dev.json',
                    format='json',
                    fields=fields)
nl.build_vocab(train_data, max_size=10000, min_freq=1)
code.build_vocab(train_data, max_size=10000, min_freq=1)

# Implementing the model
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=True)

        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        # x shape: (seq_length, N) where N is batch size

        embedding = self.embedding(x)
        # embedding shape: (seq_length, N, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding)
        # outputs shape: (seq_length, N, hidden_size)

        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        # x shape: (N) where N is for batch size, we want it to be (1, N), seq_length
        # is 1 here because we are sending in a single word and not a sentence
        x = x.unsqueeze(0)

        embedding = self.embedding(x)
        # embedding shape: (1, N, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        # outputs shape: (1, N, hidden_size)

        predictions = self.fc(outputs)

        predictions = predictions.squeeze(0)

        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(code.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        hidden, cell = self.encoder(source)

        # Grab the first input to the Decoder which will be <SOS> token
        x = target[0]

        for t in range(1, target_len):
            # Use previous hidden, cell as context from encoder at start
            output, hidden, cell = self.decoder(x, hidden, cell)

            # Store next output prediction
            outputs[t] = output

            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)

            # x = target[t]
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs



# Model hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size_encoder = len(nl.vocab)
input_size_decoder = len(code.vocab)
output_size = len(code.vocab)
encoder_embedding_size = 200
decoder_embedding_size = 200
hidden_size = 1024  # Needs to be the same for both RNN's
num_layers = 1

# load the trained model
encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers).to(device)
decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, output_size, num_layers).to(device)
model = Seq2Seq(encoder_net, decoder_net).to(device)
state_dict = torch.load("my_checkpoint.pth.tar")
model.load_state_dict(state_dict["state_dict"])

print("Evaluating ...")
# evaluate bleu score
score, targets, predictions = bleu(val_data, model, nl, code, device)
print(f"Bleu score {score:.2f}")

# evaluate exact match
exact_match = get_exact_match(targets, predictions)
print(f"Exact_match {exact_match:.2f}")


