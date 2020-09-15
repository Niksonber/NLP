
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import time
from torch import tensor, long

#plt.switch_backend('agg')

def idx2tensor(indx, device):
    return tensor(indx, dtype=long, device=device).view(-1, 1)

def tensorsFromPair(pair, device):
    input_tensor = idx2tensor(pair[0], device)
    target_tensor = idx2tensor(pair[1], device)
    return (input_tensor, target_tensor, device)
    
def asMinutes(s):
    m = s // 60
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    inp = input_sentence.split(' ') if type(input_sentence) == str else input_sentence
    # Set up axes
    ax.set_xticklabels([''] + inp +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
