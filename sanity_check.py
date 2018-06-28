#!/usr/bin/env python
from __future__ import print_function
from __future__ import unicode_literals
import sys
import argparse
import os
import subprocess
import json
import csv
import shutil
import re
import glob


def _report(code, phase, run, model):
    print('Failed sanity check for {0}: {1} - {2} model {3} not found!'.format(code, phase, run, model),
          file=sys.stderr)


def _check_sentence_segment_udpipe(treebank_conf, global_conf):
    """

    :param treebank_conf:
    :param global_conf:
    :return:
    """
    model_path = os.path.join(global_conf['model']['udpipe_model_dir'], treebank_conf['segment_model'])

    if not os.path.exists(model_path):
        _report(treebank_conf['code'], 'sentence segment', 'udpipe', model_path)


def _check_sentence_segment_uppsala(treebank_conf, global_conf):
    """

    :param treebank_conf:
    :param global_conf:
    :return:
    """
    model_path = os.path.join(global_conf['model']['sent_segment_model_dir'], treebank_conf['segment_model'])

    if not os.path.exists(model_path):
        _report(treebank_conf['code'], 'sentence segment', 'uppsala', model_path)


def _sentence_segment_thai_preprocessor(treebank_conf, global_conf):
    """

    :param treebank_conf:
    :param global_conf:
    :return:
    """
    model_path = os.path.join(global_conf['model']['sent_segment_model_dir'], treebank_conf['segment_model'])
    if not os.path.exists(model_path):
        _report(treebank_conf['code'], 'sentence segment', 'thai preprocessor', model_path)


def check_sentence_segment(treebank_conf, global_conf):
    """

    :param treebank_conf:
    :param global_conf:
    :return:
    """
    if treebank_conf['sentence_segmentor'] == 'udpipe':
        _check_sentence_segment_udpipe(treebank_conf, global_conf)
    elif treebank_conf['sentence_segmentor'] == 'uppsala':
        _check_sentence_segment_uppsala(treebank_conf, global_conf)
    elif treebank_conf['sentence_segmentor'] == 'thai_preprocessor':
        _sentence_segment_thai_preprocessor(treebank_conf, global_conf)
    else:
        print('Unknown sentence segment {}'.format(treebank_conf['sentence_segmentor']), file=sys.stderr)


def check_tokenize_aux(treebank_conf, global_conf):
    """

    :param treebank_conf:
    :param global_conf:
    :return:
    """
    if treebank_conf['tokenize_aux'] == '':
        return

    model_path = os.path.join(global_conf['model']['char_elmo'], treebank_conf['tokenize_aux_model'])
    if not os.path.exists(model_path):
        _report(treebank_conf['code'], 'tokenize aux', 'elmo', model_path)


def _check_tokenize_scir_tokenizer(treebank_conf, global_conf):
    """

    :param treebank_conf:
    :param global_conf:
    :return:
    """
    model = treebank_conf['tokenize_model']
    range_match = re.search('\[(\d)-(\d)\]', model)
    if range_match is not None:
        for i in range(int(range_match.group(1)), int(range_match.group(2)) + 1):
            model_path = os.path.join(global_conf['model']['tokenize_model_dir'],
                                      re.sub('\[\d-\d\]', str(i), model))
            if not os.path.exists(model_path):
                _report(treebank_conf['code'], 'tokenize', 'scir tokenizer', model_path)
    else:
        model_path = os.path.join(global_conf['model']['tokenize_model_dir'], model)
        if not os.path.exists(model_path):
            _report(treebank_conf['code'], 'tokenize', 'scir tokenizer', model_path)


def check_tokenize(treebank_conf, global_conf):
    """

    :param treebank_conf:
    :param global_conf:
    :return:
    """
    if treebank_conf['tokenizer'] == 'cp':
        pass
    elif treebank_conf['tokenizer'] == 'scir_tokenizer':
        _check_tokenize_scir_tokenizer(treebank_conf, global_conf)
    else:
        raise ValueError('Unknown tokenizer {0}'.format(treebank_conf['tokenizeer']))


def _check_morphology_tag_udpipe(treebank_conf, global_conf):
    """

    :param treebank_conf:
    :param global_conf:
    :return:
    """
    model_path = os.path.join(global_conf['model']['udpipe_model_dir'], treebank_conf['morphology_model'])
    if not os.path.exists(model_path):
        _report(treebank_conf['code'], 'morphology tag', 'udpipe', model_path)


def check_morphology_tag(treebank_conf, global_conf):
    """

    :param treebank_conf:
    :param global_conf:
    :return:
    """
    if treebank_conf['morphology_tagger'] == 'udpipe':
        _check_morphology_tag_udpipe(treebank_conf, global_conf)
    elif treebank_conf['morphology_tagger'] == 'cp':
        pass
    else:
        raise ValueError('Unknown tagger {0}'.format(treebank_conf['tagger']))


def check_postag_aux(treebank_conf, global_conf):
    """

    :param treebank_conf:
    :param global_conf:
    :return:
    """
    if treebank_conf['tagger_aux'] == '':
        return
    model_path = os.path.join(global_conf['model']['word_elmo'], treebank_conf['elmo_model'])
    if not os.path.exists(model_path):
        _report(treebank_conf['code'], 'postag aux', 'elmo', model_path)


def _check_postag_stanford(treebank_conf, global_conf):
    """

    :param treebank_conf:
    :param global_conf:
    :return:
    """
    model_path = os.path.join(global_conf['model']['postag_model_dir'], treebank_conf['tag_model'])
    if not os.path.exists(model_path):
        _report(treebank_conf['code'], 'postag', 'stanford', model_path)

    if treebank_conf['embeddings'] != '':
        embedding_file = os.path.join(global_conf['resources']['word_embeddings'], treebank_conf['embeddings'])
        if not os.path.exists(embedding_file):
            _report(treebank_conf['code'], 'postag', 'stanford embeddings', embedding_file)


def check_postag(treebank_conf, global_conf):
    """

    :param treebank_conf:
    :param global_conf:
    :return:
    """
    if treebank_conf['tagger'] == 'stanford':
        _check_postag_stanford(treebank_conf, global_conf)
    elif treebank_conf['tagger'] == 'cp':
        pass
    else:
        raise ValueError('Unknown tagger {0}'.format(treebank_conf['tagger']))


def check_parse_aux(treebank_conf, global_conf):
    """

    :param treebank_conf:
    :param global_conf:
    :return:
    """
    if treebank_conf['parser_aux'] == '':
        return

    model_path = os.path.join(global_conf['model']['word_elmo'], treebank_conf['elmo_model'])
    if not os.path.exists(model_path):
        _report(treebank_conf['code'], 'parse aux', 'elmo', model_path)


def _check_parse_stanford(treebank_conf, global_conf):
    """

    :param treebank_conf:
    :param global_conf:
    :return:
    """

    range_match = re.search('\[(\d)-(\d)\]', treebank_conf['parse_model'])
    if range_match is not None:
        for i in range(int(range_match.group(1)), int(range_match.group(2)) + 1):
            model_path = os.path.join(global_conf['model']['parse_model_dir'],
                                      re.sub('\[\d-\d\]', str(i), treebank_conf['parse_model']))
            if not os.path.exists(model_path):
                _report(treebank_conf['code'], 'parse', 'stanford', model_path)
    else:
        model_path = os.path.join(global_conf['model']['parse_model_dir'], treebank_conf['parse_model'])
        if not os.path.exists(model_path):
            _report(treebank_conf['code'], 'parse', 'stanford', model_path)

    if treebank_conf['embeddings'] != '':
        embedding_file = os.path.join(global_conf['resources']['word_embeddings'], treebank_conf['embeddings'])
        if not os.path.exists(embedding_file):
            _report(treebank_conf['code'], 'parser', 'stanford embeddings', embedding_file)


def _check_parse_trans_parser(treebank_conf, global_conf):
    """

    :param treebank_conf:
    :param global_conf:
    :return:
    """
    model_dir = os.path.join(global_conf['model']['parse_model_dir'], treebank_conf['parse_model'])
    info_file = os.path.join(model_dir, '{0}.infos'.format(treebank_conf['code']))
    if not os.path.exists(info_file):
        _report(treebank_conf['code'], 'parse', 'trans_parser', info_file)

    model1_file = os.path.join(model_dir, [f for f in os.listdir(model_dir) if '-v0' and '.params' in f][0])
    if not os.path.exists(model1_file):
        _report(treebank_conf['code'], 'parse', 'trans_parser', model1_file)

    model2_file = os.path.join(model_dir, [f for f in os.listdir(model_dir) if '-v1' and '.params' in f][0])
    if not os.path.exists(model2_file):
        _report(treebank_conf['code'], 'parse', 'trans_parser', model2_file)


def check_parse(treebank_conf, global_conf):
    """

    :param treebank_conf:
    :param global_conf:
    :return:
    """
    if treebank_conf['parser'] == 'stanford':
        _check_parse_stanford(treebank_conf, global_conf)
    elif treebank_conf['parser'] == 'trans_parser':
        _check_parse_trans_parser(treebank_conf, global_conf)
    else:
        raise ValueError('unknown parser {0}'.format(treebank_conf['parser']))


def load_global_conf():
    """


    :return:
    """
    root_dir, _ = os.path.split(os.path.abspath(__file__))
    conf_dir = os.path.join(root_dir, 'conf')

    conf = json.load(open(os.path.join(conf_dir, 'global.json'), 'r'))
    return conf


def load_conf():
    """

    :return:
    """
    root_dir, _ = os.path.split(os.path.abspath(__file__))
    conf_dir = os.path.join(root_dir, 'conf')
    reader = csv.DictReader(open(os.path.join(conf_dir, 'dataset.csv'), 'r'))
    output = {}
    for row in reader:
        output[row['code']] = row
    return output


def main():
    cmd = argparse.ArgumentParser('usage: ')
    cmd.add_argument('-input_dir', required=True, help='the path to the input dir.')
    opts = cmd.parse_args()

    # get global configuration
    global_conf = load_global_conf()

    dataset_conf = load_conf()

    meta = json.load(open(os.path.join(opts.input_dir, 'metadata.json')))
    for info in sorted(meta, key=lambda entry: (entry['lcode'] in ('zh', 'ja', 'vi'), entry['lcode'], entry['tcode']),
                       reverse=True):
        code = '{0}_{1}'.format(info['lcode'], info['tcode'])
        print('checking {0} ...'.format(code), file=sys.stderr)
        treebank_conf = dataset_conf[code]

        # sentence segmentation
        check_sentence_segment(treebank_conf, global_conf)

        # tokenize aux
        check_tokenize_aux(treebank_conf, global_conf)

        # tokenize
        check_tokenize(treebank_conf, global_conf)

        # morphology_tag
        check_morphology_tag(treebank_conf, global_conf)

        # postag aux
        check_postag_aux(treebank_conf, global_conf)

        # postag
        check_postag(treebank_conf, global_conf)

        # parse aux
        check_parse_aux(treebank_conf, global_conf)

        # parse
        check_parse(treebank_conf, global_conf)


if __name__ == "__main__":
    main()
