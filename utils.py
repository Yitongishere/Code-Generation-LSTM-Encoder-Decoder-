import torch
import spacy
import numpy as np
from torchtext.data.metrics import bleu_score
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


def translate_sentence(model, sentence, nl, code, device, max_length=100):

    # Load nl tokenizer
    spacy_nl = spacy.load("en_core_web_sm")

    # Create tokens using spacy
    if type(sentence) == str:
        tokens = [token.text for token in spacy_nl(sentence)]
    else:
        tokens = [token for token in sentence]

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, nl.init_token)
    tokens.append(nl.eos_token)

    # Go through each token and convert to an index
    text_to_indices = [nl.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    # Build encoder hidden, cell state
    with torch.no_grad():
        hidden, cell = model.encoder(sentence_tensor)

    outputs = [code.vocab.stoi["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == code.vocab.stoi["<eos>"]:
            break

    translated_sentence = [code.vocab.itos[idx] for idx in outputs]

    # remove start token
    return translated_sentence[1:]


def bleu(data, model, nl, code, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["nl"]
        trg = vars(example)["code"]

        prediction = translate_sentence(model, src, nl, code, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets) * 100, targets, outputs

def get_exact_match(targets, predictions):
    num_exact_match = 0

    for i in range(len(targets)):
        if targets[i][0] == predictions[i][:len(targets[i][0])]:
            num_exact_match += 1

    exact_match = float(num_exact_match) / len(targets) * 100

    return exact_match


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])



def plot_history(history, num_epoch, file_name="./train_history.png"):
    """

    Args:
        history: a dictionary

    Returns:
        Training history figure
    """
    fig, axs = plt.subplots(3, 1, figsize=(12,15))
    epoch = np.arange(1, num_epoch+1, 1)
    x_major_locator = MultipleLocator(1)
    y_major_locator = MultipleLocator(5)
    y_major_locator1 = MultipleLocator(0.5)

    # create loss plot
    axs[0].plot(epoch, history['loss_history'], label="Train Loss")
    axs[0].plot(epoch, history['val_loss_history'], label="Val Loss")
    axs[0].set_ylabel("Loss")
    axs[0].set_xlabel("epoch")
    axs[0].legend(loc="upper right")
    axs[0].set_title("Loss eval")
    axs[0].xaxis.set_major_locator(x_major_locator)
    axs[0].yaxis.set_major_locator(y_major_locator1)
    axs[0].set_ylim(-0.02, 4)

    # create bleu plot
    axs[1].plot(epoch, history['bleu_history'], label="Train Bleu")
    axs[1].plot(epoch, history['val_bleu_history'], label="Val Bleu")
    axs[1].set_ylabel("Bleu")
    axs[1].set_xlabel("epoch")
    axs[1].legend(loc="lower right")
    axs[1].set_title("Bleu eval")
    axs[1].xaxis.set_major_locator(x_major_locator)
    axs[1].yaxis.set_major_locator(y_major_locator)
    axs[1].set_ylim(-0.02, 50)

    # create exact-match plot
    axs[2].plot(epoch, history['exact_match_history'], label="Train Exact match")
    axs[2].plot(epoch, history['val_exact_match_history'], label="Val Exact match")
    axs[2].set_ylabel("Exact match")
    axs[2].set_xlabel("epoch")
    axs[2].legend(loc="lower right")
    axs[2].set_title("Exact match eval")
    axs[2].xaxis.set_major_locator(x_major_locator)
    axs[2].yaxis.set_major_locator(y_major_locator1)
    axs[2].set_ylim(-0.02, 5)

    plt.savefig(file_name)
    # plt.show()