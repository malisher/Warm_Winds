import collections.abc as collections
from copy import copy
import torch


class StrLabelConverter(object):

    def __init__(self, alphabet, ignore_case=True):

        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = '-' + alphabet

        self.alphabet_indicies = {char: i for i, char in enumerate(self.alphabet)}

    def encode(self, text):
        original_text = copy(text)
        #print(text)
        try:
            if isinstance(text, str):
                text = text.replace(',', '').replace('`','')
                text = [self.alphabet_indicies[char.lower() if self._ignore_case else char] for char in text]
                length = [len(text)]
            elif isinstance(text, collections.Iterable):
                length = [len(s) for s in text]
                text = ''.join(text)
                text, _ = self.encode(text)
            return torch.LongTensor(text), torch.LongTensor(length)
        except Exception as E:
            print('Error is here')
            print(E)
            print(original_text)
            return None

    def decode(self, t, length, raw=False):

        if length.numel() == 1:
            length = length.item()
            if raw:
                return ''.join([self.alphabet[i] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i] == t[i - 1])):
                        char_list.append(self.alphabet[t[i]])
                return ''.join(char_list)
        else:
            # batch mode
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(self.decode(t[:, i], torch.IntTensor([l]), raw=raw))
                index += l
            return texts
