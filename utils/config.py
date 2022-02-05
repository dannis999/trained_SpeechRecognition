import json

_pinyin_dict = None
_pinyin_list = None

def load_pinyin_dict(filename: str) -> tuple:
    '''
    加载拼音列表和拼音字典

    拼音列表：用于下标索引转拼音 \\
    拼音字典：用于拼音索引转下标
    '''
    global _pinyin_list, _pinyin_dict
    if _pinyin_dict is not None and _pinyin_list is not None:
        return _pinyin_list, _pinyin_dict

    _pinyin_list = list()
    _pinyin_dict = dict()
    with open(filename, 'r', encoding='utf-8') as file_pointer:
        lines = file_pointer.read().split('\n')
    for line in lines:
        if len(line) == 0:
            continue
        tokens = line.split('\t')
        _pinyin_list.append(tokens[0])
        _pinyin_dict[tokens[0]] = len(_pinyin_list) - 1
    return _pinyin_list, _pinyin_dict
