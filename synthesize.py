import torch
import numpy as np
import os
import argparse
import librosa
import re
import json
from string import punctuation
from g2p_en import G2p

from models.StyleSpeech import StyleSpeech
from models.Wav2vec2 import Wav2vec2
from text import text_to_sequence
import audio as Audio
import utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, lexicon_path):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(lexicon_path)

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(text_to_sequence(phones, ['english_cleaners']))

    return torch.from_numpy(sequence).to(device=device)


def preprocess_audio(audio_file, _stft):
    wav, sample_rate = librosa.load(audio_file, sr=None)
    wav_ = wav
    if sample_rate != 22050:
        wav = librosa.resample(wav, sample_rate, 22050)
    '''
    In Zero-shot of VCTK dataset, it is effective to trim the silence (top_db=15)
    '''
    wav, section = librosa.effects.trim(wav, top_db=15)
    wav16 = librosa.resample(wav, sample_rate, 16000)
    wav16 = torch.FloatTensor(wav16).unsqueeze(0)
    
    mel_spectrogram, _ = Audio.tools.get_mel_from_wav(wav, _stft)
    return wav16.to(device), torch.from_numpy(mel_spectrogram).to(device=device)


def get_StyleSpeech(config, checkpoint_path):
    model = StyleSpeech(config).to(device=device)
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    model.eval()
    return model


def synthesize(args, text, model, wav2vec2, _stft):   
    # preprocess audio and text
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    wav16, ref_mel = preprocess_audio(args.ref_audio, _stft)
    ref_mel = ref_mel.transpose(0,1).unsqueeze(0)
    mel_ref_ = ref_mel.cpu().squeeze().transpose(0, 1).detach()
    np.save(save_path + 'ref_{}.npy'.format(args.ref_audio[-12:-4]), np.array(mel_ref_.unsqueeze(0)))

    #* (1, T, 756)
    latent = wav2vec2(wav16)
    print('latent.shape', latent.shape)
    # Extract style vector
    # style_vector = model.get_style_vector(ref_mel)
    #* (1, 1, 128)
    style_vector = model.get_style_vector(latent)
    print('style_vector.shape', style_vector.shape)
    
    if isinstance(text, list):
        for txt in text:
            src = preprocess_english(txt, args.lexicon_path).unsqueeze(0)
            src_len = torch.from_numpy(np.array([src.shape[1]])).to(device=device)
            # Forward
            mel_output = model.inference(style_vector, src, src_len)[0]
            mel_ = mel_output.cpu().squeeze().transpose(0, 1).detach()
            np.save(save_path + '{}.npy'.format(txt[:10]), np.array(mel_.unsqueeze(0)))

    else:
        src = preprocess_english(text, args.lexicon_path).unsqueeze(0)
        src_len = torch.from_numpy(np.array([src.shape[1]])).to(device=device)
        # Forward
        mel_output = model.inference(style_vector, src, src_len)[0]
        mel_ = mel_output.cpu().squeeze().transpose(0, 1).detach()
        np.save(save_path + '{}.npy'.format(text[:10]), np.array(mel_.unsqueeze(0)))

    print('Generate done!')

    # plotting
    # utils.plot_data([mel_ref_.numpy(), mel_.numpy()], 
    #     ['Ref Spectrogram', 'Synthesized Spectrogram'], filename=os.path.join(save_path, 'plot.png'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, 
        help="Path to the pretrained model")
    parser.add_argument('--config', default='configs/config.json')
    parser.add_argument("--save_path", type=str, default='results/')
    parser.add_argument("--ref_audio", type=str, required=True,
        help="path to an reference speech audio sample")
    parser.add_argument("--text", type=str, default = 'In being comparatively modern.',
        help="raw text to synthesize")
    parser.add_argument("--lexicon_path", type=str, default='lexicon/librispeech-lexicon.txt')
    args = parser.parse_args()
    
    with open(args.config) as f:
        data = f.read()
    json_config = json.loads(data)
    config = utils.AttrDict(json_config)

    # Get model
    model = get_StyleSpeech(config, args.checkpoint_path)
    wav2vec2 = Wav2vec2().to(device)
    wav2vec2.eval()
    print('model is prepared')

    _stft = Audio.stft.TacotronSTFT(
                config.filter_length,
                config.hop_length,
                config.win_length,
                config.n_mel_channels,
                config.sampling_rate,
                config.mel_fmin,
                config.mel_fmax)

    # Synthesize
    args.text = [
        'but which must certainly have come from the study of the twelfth or even the eleventh century MSS.',
        'many of whose types, indeed, like that of the Subiaco works, are of a transitional character.',
        "instead of ending in the sharp and clear stroke of Jenson's letters;",
        'modern printers generally overdo the "whites" in the spacing, a defect probably forced on them by the characterless quality of the letters.',
        # 'with criminals and misdemeanants of all shades crowding perpetually into its narrow limits, the latter state of Newgate was worse than the first.',
        'Other cases are recorded elsewhere, as at the Giltspur Street Compter, where in eighteen oh five Mr. Neild found a man named William Grant',
        'the Fleet, and the Marshalsea prisons especially devoted to them,',
        # 'whose net annual income thus entirely derived from the impecunious amounted to between three and four thousand pounds.',
        'Matters were rather better at the Marshalsea.',
        # 'and denied admission to the "charity wards," which partook of all the benefits of bequests and donations to poor debtors.',
        # "on the master's side it was thirteen and fourpence, and a gallon of beer on entrance, although Mr. Newman,",
        # 'no steps taken to reduce the number of committals, and the governor was obliged to utilize the chapel as a day and night room.',
        "Authority was given to raise money on the orphans' Fund to the extent of ninety thousand pounds.",
        'which would accommodate the inmates of Newgate and of the three compters, Ludgate,',
        'was frequently adjudged for so-called libels, or too out-spoken comments in print.',
        # 'that some other and less mixed prison should be used for the confinement of persons convicted of libels. But this suggestion was ignored.',
        # 'whose language and manners, whose female associates of the most abandoned description, and the scenes consequent with such lost wretches',
        # 'from whom still higher fees were exacted, with the same discreditable idea of swelling the revenues of the prison.',
        # "the misdemeanant tried or untried, the debtor who wished to avoid the discomfort of the crowded debtors' side, the outspoken newspaper editor,",
        'A further and a more iniquitous method of extorting money',
        # 'the surgeon should see all prisoners, whether ill or well, once a week, and take general charge of the infirmaries.',
        'gaming of all sorts should be peremptorily forbidden under heavy pains and penalties.',
        'The crowding was in consequence of the delay in removing transports.',
        # 'These often remained in Newgate for six months, sometimes a year, in some cases longer;',
        # 'in one, for seven years -- that of a man sentenced to death, for whom great interest had been made, but whom it was not thought right to pardon.',
        # 'Occasionally the transports made themselves so useful in the jail that they were passed over.',
        # 'Mr. Newman admitted that he had petitioned that certain "trusty men" might be left in the jail.',
        # 'Constantly associated with these convicted felons were numbers of juveniles, infants of tender years.',
        # 'There were frequently in the middle yard seven or eight children, the youngest barely nine,',
        # 'the oldest only twelve or thirteen, exposed to all the contaminating influences of the place.',
        # 'Mr. Bennet mentions also the case of young men of better stamp, clerks in city offices, and youths of good parentage,',
        # 'Quote, in this dreadful situation, end quote. who had been rescued from the hulks through the kindness and attention of the Secretary of State.',
        # 'Quote, yet they had been long enough, he goes on to say, in the prison associated with the lowest and vilest criminals,',
        # 'with convicts of all ages and characters, to render it next to impossible but that, with the obliteration of all sense of self-respect,',
        # 'the inevitable consequence of such a situation, their morals must have been destroyed;',
        # 'and though distress or the seduction of others might have led to the commission of this their first offense,',
        # 'yet the society they were driven to live in, the language they daily heard, and the lessons they were taught in this academy,',
        # 'must have had a tendency to turn them into the world hardened and accomplished in the ways of vice and crime. End quote.',
        # 'Mr. Buxton, in the work already quoted, instances another grievous case of the horrors of indiscriminate association in Newgate.',
        # 'It was that of a person, quote, who practiced in the law, and who was connected by marriage with some very respectable families.',
        'Having been committed to Clerkenwell,',
        # 'he was sent on to Newgate in a coach, handcuffed to a noted house-breaker, who was afterwards cast for death.',
        # 'The first night in Newgate, and for the subsequent fortnight,',
        # 'he slept in the same bed with a highwayman on one side, and a man charged with murder on the other.',
        # 'Spirits were freely introduced, and although he at first abstained,',
        # 'he found he must adopt the manners of his companions, or that his life would be in danger.',
        # 'They viewed him with some suspicion, as one of whom they knew nothing.',
        # 'He was in consequence put out of the protection of their internal law, end quote. Their code was a subject of some curiosity.',
        # 'When any prisoner committed an offense against the community or against an individual, he was tried by a court in the jail.',
        # 'A prisoner, generally the oldest and most dexterous thief,',
        # 'was appointed judge, and a towel tied in knots was hung on each side in imitation of a wig.',
        # 'The judge sat in proper form; he was punctiliously styled "my lord."',
        # 'A jury having been selected and duly sworn, the culprit was then arraigned. Justice, however, was not administered with absolute integrity.',
        # 'A bribe to the judge was certain to secure acquittal, and the neglect of the formality was as certainly followed by condemnation.',
        # 'Various punishments were inflicted, the heaviest of which was standing in the pillory.',
        # "This was carried out by putting the criminal's head through the legs of a chair, and stretching out his arms and tying them to the legs.",
        # 'The culprit was then compelled to carry the chair about with him.',
        # 'But all punishments might readily be commuted into a fine to be spent in gin for judge and jury.',
        # 'The prisoner mentioned above was continually persecuted by trials of this kind.',
        'The most trifling acts were magnified into offenses.',
        # 'He was charged with moving something which should not be touched, with leaving a door open, or coughing maliciously to the disturbance of his companions.',
        # 'The evidence was invariably sufficient to convict, and the judge never hesitated to inflict the heaviest penalties.',
        # 'The unfortunate man was compelled at length to adopt the habits of his associates;',
        # 'Quote, by insensible degrees he began to lose his repugnance to their society,',
        # 'caught their flash terms and sung their songs, was admitted to their revels, and acquired, in place of habits of perfect sobriety,',
        'a taste for spirits. End quote.',
        # 'His wife visited him in Newgate, and wrote a pitiable account of the state in which she found her husband.',
        # 'He was an inmate of the same ward with others of the most dreadful sort, quote,',
        # 'whose language and manners, whose female associates of the most abandoned description, and the scenes consequent with such lost wretches',
        # 'prevented me from going inside but seldom, and I used to communicate with him through the bars from the passage. End quote.',
        'One day he was too ill to come down and meet her.',
        'She went up to the ward and found him lying down, quote,',

    ]

    synthesize(args, args.text, model, wav2vec2, _stft)