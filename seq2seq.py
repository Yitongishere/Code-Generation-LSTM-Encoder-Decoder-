import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, TabularDataset, BucketIterator
import spacy
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint, plot_history, get_exact_match
import random

# tokenizer
tokenize_code = lambda x: x.split()
spacy_eng = spacy.load("en_core_web_sm")
def tokenize_nl(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

# setup the Dataset
nl = Field(tokenize=tokenize_nl, lower=False, init_token="<sos>", eos_token="<eos>")
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

        # predictions shape: (1, N, length_target_vocabulary) to send it to loss function
        # we want it to be (N, length_target_vocabulary) so we're just gonna remove the first dim
        predictions = predictions.squeeze(0)

        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.7):
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

            # With probability of teacher_force_ratio we take the actual next word otherwise we take the word that the Decoder predicted it to be.
            # x = target[t]
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs

# Training hyperparameters
num_epochs = 20
learning_rate = 0.00003
batch_size = 64

# Model hyperparameters
load_model = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size_encoder = len(nl.vocab)
input_size_decoder = len(code.vocab)
output_size = len(code.vocab)
encoder_embedding_size = 200
decoder_embedding_size = 200
hidden_size = 1024  # Needs to be the same for both RNN's
num_layers = 1

# Data loader
train_iterator, val_iterator = BucketIterator.splits(
    (train_data, val_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.nl),
    device=device,
)

encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers).to(device)
decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, output_size, num_layers).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = code.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

# an Example semtence for observing training processing
sentence = "comment f1z concode_field_sep int I1z concode_elem_sep int I2z concode_elem_sep int I1 concode_elem_sep int I3 concode_elem_sep int I2 concode_elem_sep int I4 concode_field_sep void fz concode_elem_sep void f concode_elem_sep void f11 concode_elem_sep void f1 concode_elem_sep void f2"

# empty lists for recoding training history for plotting learning curve
loss_history = []
bleu_history = []
exact_match_history = []
val_loss_history = []
val_bleu_history = []
val_exact_match_history = []


for epoch in range(1, num_epochs+1):
    print(f"[Epoch {epoch} / {num_epochs}]")

    sum_loss = []
    for batch_idx, batch in enumerate(train_iterator):
        print(f"\r(batch {batch_idx+1} / {len(train_iterator)})", end='')
        # Get input and targets and get to cuda
        inp_data = batch.nl.to(device)
        target = batch.code.to(device)

        # Forward prop
        output = model(inp_data, target)

        # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshaping. While we're at it
        # Let's also remove the start token while we're at it
        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)
        sum_loss.append(loss)

        # Back prop
        loss.backward()

        # Clip to avoid exploding gradient issues, makes sure grads are
        # within a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Gradient descent step
        optimizer.step()

    # training loss
    mean_loss = float(sum(sum_loss)) / len(sum_loss)
    loss_history.append(mean_loss)
    print(f"\nTraining Loss: {mean_loss:.2f} ")
    # training bleu
    bleu_score, targets, predictions = bleu(train_data[:2000], model, nl, code, device)
    bleu_history.append(bleu_score)
    print(f"Training Bleu score {bleu_score:.2f} ")
    # training exact_match
    exact_match = get_exact_match(targets, predictions)
    exact_match_history.append(exact_match)
    print(f"Training Exact match {exact_match:.2f} ")

    # evaluate after every epoch
    model.eval()
    with torch.no_grad():
        print("Validating ... ")
        val_sum_loss = []
        for batch_idx, batch in enumerate(val_iterator):
            inp_data = batch.nl.to(device)
            target = batch.code.to(device)
            output = model(inp_data, target)
            output = output[1:].reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)
            val_loss = criterion(output, target)
            val_sum_loss.append(val_loss)
        # validation loss
        val_mean_loss = float(sum(val_sum_loss)) / len(val_sum_loss)
        val_loss_history.append(val_mean_loss)
        print(f"Validation Loss: {val_mean_loss:.2f} ")
        # validation bleu
        val_bleu_score, val_targets, val_predictions = bleu(val_data, model, nl, code, device)
        val_bleu_history.append(val_bleu_score)
        print(f"Validation Bleu score {val_bleu_score:.2f} ")
        # validation exact_match
        val_exact_match = get_exact_match(val_targets, val_predictions)
        val_exact_match_history.append(val_exact_match)
        print(f"Validation Exact match {val_exact_match:.2f} ")

        # check the translated example sentence
        translated_sentence = translate_sentence(model, sentence, nl, code, device, max_length=100)
        print(f"Translated example sentence: \n {translated_sentence}")

    model.train()

    # save the model
    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    save_checkpoint(checkpoint)

    # recoding the training process
    history = {'loss_history': loss_history,
               'val_loss_history': val_loss_history,
               'bleu_history': bleu_history,
               'val_bleu_history': val_bleu_history,
               'exact_match_history': exact_match_history,
               'val_exact_match_history': val_exact_match_history}

    plot_history(history, epoch, file_name=f"./train_history_epoch{epoch}.png")





