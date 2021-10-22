import numpy as np
import torch

from text import text_to_sequence
from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT
from train import load_model
#from text import text_to_sequence
from audio_processing import griffin_lim
from utils import load_wav_to_torch
from scipy.io.wavfile import write
import os
import time

from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
#import IPython.display as ipd
#from tqdm import tqdm

hparams = create_hparams()
hparams.sampling_rate = 22050
hparams.max_decoder_steps = 1000

audio_path = 'griffin_lim/checkpoint_33500_3'
extension = '.wav'
save_path = audio_path + extension


stft = TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)

def load_mel(path):
    audio, sampling_rate = load_wav_to_torch(path)
    if sampling_rate != hparams.sampling_rate:
        raise ValueError("{} SR doesn't match target {} SR".format(
            sampling_rate, stft.sampling_rate))
    audio_norm = audio / hparams.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = melspec.cuda()
    return melspec

def plot_data_aligment(data, figsize=(16, 4)):
   # fig, axes = plt.subplots(1, len(data), figsize=figsize)
    plt.imshow(data, aspect='auto', origin='bottom', interpolation='none')
    plt.savefig('./griffin_lim/aligment_33500_ref23jpg')


def plot_data(data, figsize=(16, 4)):
    plt.figure(figsize=figsize)
    plt.imshow(data, aspect='auto', origin='bottom', interpolation='none')
    plt.savefig('./griffin_lim/33500_ref3.jpg')

checkpoint_path = "./outdir/checkpoint_33500"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.eval()

def TextEncoder(text):
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    inputs = model.parse_input(sequence)
    embedded_inputs = model.embedding(inputs).transpose(1,2)
    transcript_outputs = model.encoder.inference(embedded_inputs)
    
    return transcript_outputs


def Decoder(encoder_outputs):
    decoder_input = model.decoder.get_go_frame(encoder_outputs)
    model.decoder.initialize_decoder_states(encoder_outputs, mask=None)
    mel_outputs, gate_outputs, alignments = [], [], []

    while True:
        decoder_input = model.decoder.prenet(decoder_input)
        mel_output, gate_output, alignment = model.decoder.decode(decoder_input)

        mel_outputs += [mel_output]
        gate_outputs += [gate_output]
        alignments += [alignment]

        if torch.sigmoid(gate_output.data) > hparams.gate_threshold:
            print(torch.sigmoid(gate_output.data), gate_output.data)
            break
        if len(mel_outputs) == hparams.max_decoder_steps:
            print("Warning! Reached max decoder steps")
            break

        decoder_input = mel_output

    mel_outputs, gate_outputs, alignments = model.decoder.parse_decoder_outputs(
        mel_outputs, gate_outputs, alignments)
    mel_outputs_postnet = model.postnet(mel_outputs)
    mel_outputs_postnet = mel_outputs + mel_outputs_postnet

    plot_data_aligment(alignments.float().data.cpu().numpy()[0].T) 
        
    return mel_outputs_postnet



def generate_mels_by_ref_audio(text, ref_audio):
    transcript_outputs = TextEncoder(text)
    print("ref_audio")
   # ipd.display(ipd.Audio(ref_audio, rate=hparams.sampling_rate))
    ref_audio_mel = load_mel(ref_audio)
    latent_vector = model.gst(ref_audio_mel)
    latent_vector = latent_vector.expand_as(transcript_outputs)

    encoder_outputs = transcript_outputs + latent_vector
    
    mel_outputs = Decoder(encoder_outputs)
    
    return mel_outputs

text = "Baby,I love you very much!"

ref_wav='./res/ref3.wav'
mel_outputs_postnet = generate_mels_by_ref_audio(text, ref_wav)
print(mel_outputs_postnet.size())
#print("text2mel synthesis successfully performed.")
plot_data(mel_outputs_postnet.data.cpu().numpy()[0])

#test_mel = "griffin_lim.npy"
#mel_dir = './griffin_lim'
#with torch.no_grad():
np.save("test_mel.npy", mel_outputs_postnet.cpu().detach().numpy(), allow_pickle = False)


# Griffin Lim vocoder synthesis:

griffin_iters = 60

mel_decompress = stft.spectral_de_normalize(mel_outputs_postnet)
mel_decompress = mel_decompress.transpose(1, 2).data.cpu()

spec_from_mel_scaling = 1000
spec_from_mel = torch.mm(mel_decompress[0], stft.mel_basis)
spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
spec_from_mel = spec_from_mel * spec_from_mel_scaling

audio = griffin_lim(torch.autograd.Variable(spec_from_mel[:, :, :-1]), stft.stft_fn, griffin_iters)

audio = audio.squeeze()
audio_numpy = audio.cpu().numpy()

write(save_path, 22050, audio_numpy)

print("mel2audio synthesis successfully performed.")

