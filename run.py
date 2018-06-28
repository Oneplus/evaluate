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


def _sentence_segment_udpipe(treebank_conf, global_conf, input_file, output_file):
    """

    :param treebank_conf:
    :param global_conf:
    :param input_file: str
    :param output_file: str
    :return:
    """
    model_path = os.path.join(global_conf['model']['udpipe_model_dir'], treebank_conf['segment_model'])
    cmds = global_conf['exec']['udpipe'] + ['-tokenize', '-tag', '-parse', model_path, input_file]

    print('running udpipe commands for {}'.format(treebank_conf['code']), file=sys.stderr)
    print(' '.join(cmds), file=sys.stderr)
    pipe = subprocess.Popen(cmds, stdout=open(output_file, 'w'))
    pipe.wait()


def _sentence_segment_uppsala(treebank_conf, global_conf, input_file, output_file):
    """

    :param treebank_conf:
    :param global_conf:
    :param input_file:
    :param output:
    :return:
    """
    model_path = os.path.join(global_conf['model']['sent_segment_model_dir'], treebank_conf['segment_model'])
    cmds = global_conf['exec']['uppsala_segmentor'] + ['tag', '-ens', '-p', model_path, '-m', treebank_conf['code'],
                                                       '-r', input_file, '-opth', output_file]
    print('running uppsala segment commands for {}'.format(treebank_conf['code']), file=sys.stderr)
    print(' '.join(cmds), file=sys.stderr)
    pipe = subprocess.Popen(cmds)
    pipe.wait()


def _sentence_segment_thai_preprocessor(treebank_conf, global_conf, input_file, output_file):
    """

    :param treebank_conf:
    :param global_conf:
    :param input_file:
    :param output_file:
    :return:
    """
    model_path = os.path.join(global_conf['model']['sent_segment_model_dir'], treebank_conf['segment_model'])
    cmds = global_conf['exec']['thai_preprocessor'] + ['--input', input_file, '--output', output_file, '--dict', model_path]

    print('running thai preprocess for {}'.format(treebank_conf['code']), file=sys.stderr)
    print(' '.join(cmds), file=sys.stderr)
    pipe = subprocess.Popen(cmds)
    pipe.wait()


def sentence_segment(treebank_conf, global_conf, input_dir, info):
    """

    :param treebank_conf:
    :param global_conf:
    :param input_dir:
    :param info:
    :return:
    """
    input_file = os.path.join(input_dir, info['rawfile'])
    output_file = os.path.join(global_conf['output'], 'segmented.conllu')

    if treebank_conf['sentence_segmentor'] == 'udpipe':
        _sentence_segment_udpipe(treebank_conf, global_conf, input_file, output_file)
    elif treebank_conf['sentence_segmentor'] == 'uppsala':
        _sentence_segment_uppsala(treebank_conf, global_conf, input_file, output_file)
    elif treebank_conf['sentence_segmentor'] == 'thai_preprocessor':
        _sentence_segment_thai_preprocessor(treebank_conf, global_conf, input_file, output_file)
    else:
        print('Unknown sentence segment {}'.format(treebank_conf['sentence_segmentor']), file=sys.stderr)


def tokenize_aux(treebank_conf, global_conf):
    """

    :param treebank_conf:
    :param global_conf:
    :return:
    """
    if treebank_conf['tokenize_aux'] == '':
        return

    input_file = os.path.join(global_conf['output'], 'segmented.conllu')
    ave_output_file = os.path.join(global_conf['output'], 'chars.ave_elmo')
    lstm_output_file = os.path.join(global_conf['output'], 'chars.lstm_elmo')

    model_path = os.path.join(global_conf['model']['char_elmo'], treebank_conf['tokenize_aux_model'])
    input_format = 'conll_char_vi' if treebank_conf['code'].split('_')[0] == 'vi' else 'conll_char'
    cmds = global_conf['exec']['elmo'] + ['test', '--model', model_path, '--input_format', input_format,
                                          '--input', input_file, '--output_ave', ave_output_file,
                                          '--output_lstm', lstm_output_file]

    print('running tokenize_aux commands for {}'.format(treebank_conf['code']), file=sys.stderr)
    print(' '.join(cmds), file=sys.stderr)
    pipe = subprocess.Popen(cmds)
    pipe.wait()


def _tokenize_scir_tokenizer(treebank_conf, global_conf, input_file, output_file):
    """

    :param treebank_conf:
    :param global_conf:
    :param input_file:
    :param output_file:
    :return:
    """
    model = treebank_conf['tokenize_model']
    range_match = re.search('\[(\d)-(\d)\]', model)
    if range_match is not None:
        model_payloads = []
        for i in range(int(range_match.group(1)), int(range_match.group(2)) + 1):
            model_payloads.append(os.path.join(global_conf['model']['tokenize_model_dir'],
                                               re.sub('\[\d-\d\]', str(i), model)))
        print(range_match.group())
        models = ','.join(model_payloads)
    else:
        models = os.path.join(global_conf['model']['tokenize_model_dir'], model)

    cmds = global_conf['exec']['scir_tokenizer'] + ['test', '--input', input_file, '--output', output_file,
                                                    '--models', models, '--use_elmo', '--test_elmo_path',
                                                    os.path.join(global_conf['output'], 'chars.ave_elmo')]
    print('running tokenize commands for {}'.format(treebank_conf['code']), file=sys.stderr)
    print(' '.join(cmds), file=sys.stderr)
    pipe = subprocess.Popen(cmds)
    pipe.wait()


def _tokenize_copy(input_file, output_file):
    """

    :param input_file:
    :param output_file:
    :return:
    """
    shutil.copy(input_file, output_file)


def tokenize(treebank_conf, global_conf):
    """

    :param treebank_conf:
    :param global_conf:
    :return:
    """
    input_file = os.path.join(global_conf['output'], 'segmented.conllu')
    output_file = os.path.join(global_conf['output'], 'tokenized.conllu')

    try:
        if treebank_conf['tokenizer'] == 'cp':
            _tokenize_copy(input_file, output_file)
        elif treebank_conf['tokenizer'] == 'scir_tokenizer':
            _tokenize_scir_tokenizer(treebank_conf, global_conf, input_file, output_file)
        else:
            raise ValueError('Unknown tokenizer {0}'.format(treebank_conf['tokenizeer']))
    except Exception:
        print('Tokenize failed, downgrade to copy', file=sys.stderr)
        _tokenize_copy(input_file, output_file)


def _morphology_tag_udpipe(treebank_conf, global_conf, input_file, output_file):
    """

    :param treebank_conf:
    :param global_conf:
    :param input_file:
    :param output_file:
    :return:
    """
    model_path = os.path.join(global_conf['model']['udpipe_model_dir'], treebank_conf['morphology_model'])
    cmds = global_conf['exec']['udpipe'] + ['-tag', '-parse', model_path, input_file]

    print('running udpipe morphology tag commands for {}'.format(treebank_conf['code']), file=sys.stderr)
    print(' '.join(cmds), file=sys.stderr)
    pipe = subprocess.Popen(cmds, stdout=open(output_file, 'w'))
    pipe.wait()


def _morphology_tag_copy(input_file, output_file):
    """

    :param input_file:
    :param output_file:
    :return:
    """
    shutil.copy(input_file, output_file)


def morphology_tag(treebank_conf, global_conf):
    """

    :param treebank_conf:
    :param global_conf:
    :return:
    """
    input_file = os.path.join(global_conf['output'], 'tokenized.conllu')
    output_file = os.path.join(global_conf['output'], 'morphology_tagged.conllu')

    try:
        if treebank_conf['morphology_tagger'] == 'udpipe':
            _morphology_tag_udpipe(treebank_conf, global_conf, input_file, output_file)
        elif treebank_conf['morphology_tagger'] == 'cp':
            _morphology_tag_copy(input_file, output_file)
        else:
            raise ValueError('Unknown tagger {0}'.format(treebank_conf['tagger']))
    except Exception:
        print('Morphology tag failed, downgrade to copy', file=sys.stderr)
        _morphology_tag_copy(input_file, output_file)


def postag_aux(treebank_conf, global_conf):
    """

    :param treebank_conf:
    :param global_conf:
    :return:
    """
    if treebank_conf['tagger_aux'] == '':
        return

    input_file = os.path.join(global_conf['output'], 'morphology_tagged.conllu')
    ave_output_file = os.path.join(global_conf['output'], 'words.ave_elmo')
    lstm_output_file = os.path.join(global_conf['output'], 'words.lstm_elmo')

    model_path = os.path.join(global_conf['model']['word_elmo'], treebank_conf['elmo_model'])
    cmds = global_conf['exec']['elmo'] + ['test', '--model', model_path, '--input_format', 'conll',
                                          '--input', input_file, '--output_ave', ave_output_file,
                                          '--output_lstm', lstm_output_file]

    print('running postag commands for {}'.format(treebank_conf['code']), file=sys.stderr)
    print(' '.join(cmds), file=sys.stderr)
    pipe = subprocess.Popen(cmds)
    pipe.wait()


def _postag_stanford(treebank_conf, global_conf, input_file, output_file):
    """

    :param treebank_conf:
    :param global_conf:
    :param input_file:
    :param output_file:
    :return:
    """
    model_path = os.path.join(global_conf['model']['postag_model_dir'], treebank_conf['tag_model'])

    cmds = global_conf['exec']['stanford'] + ['--save_dir', model_path, 'parse', input_file,
                                              '--output_file', output_file]
    if treebank_conf['embeddings'] != '':
        cmds.append('--pretrained_vocab')
        cmds.append('filename={0}'.format(os.path.join(global_conf['resources']['word_embeddings'],
                                                       treebank_conf['embeddings'])))

    _, model_dir = os.path.split(model_path)
    elmo_type = model_dir.split('-')[1]
    if elmo_type == 'lstm':
        cmds.append('--elmo_file')
        cmds.append(os.path.join(global_conf['output'], 'words.lstm_elmo'))
    elif elmo_type == 'ave':
        cmds.append('--elmo_file')
        cmds.append(os.path.join(global_conf['output'], 'words.ave_elmo'))

    print('running postag commands for {}'.format(treebank_conf['code']), file=sys.stderr)
    print(' '.join(cmds), file=sys.stderr)
    pipe = subprocess.Popen(cmds)
    pipe.wait()


def _postag_copy(input_file, output_file):
    """

    :param input_file:
    :param output_file:
    :return:
    """
    shutil.copy(input_file, output_file)


def postag(treebank_conf, global_conf):
    """

    :param treebank_conf:
    :param global_conf:
    :return:
    """
    input_file = os.path.join(global_conf['output'], 'morphology_tagged.conllu')
    output_file = os.path.join(global_conf['output'], 'tagged.conllu')

    try:
        if treebank_conf['tagger'] == 'stanford':
            _postag_stanford(treebank_conf, global_conf, input_file, output_file)
        elif treebank_conf['tagger'] == 'cp':
            _postag_copy(input_file, output_file)
        else:
            raise ValueError('Unknown tagger {0}'.format(treebank_conf['tagger']))
    except Exception:
        print('Postag failed, downgrade to copy', file=sys.stderr)
        _postag_copy(input_file, output_file)


def parse_aux(treebank_conf, global_conf):
    """

    :param treebank_conf:
    :param global_conf:
    :return:
    """
    if treebank_conf['parser_aux'] == '':
        return

    input_file = os.path.join(global_conf['output'], 'tokenized.conllu')
    ave_output_file = os.path.join(global_conf['output'], 'words.ave_elmo')
    lstm_output_file = os.path.join(global_conf['output'], 'words.lstm_elmo')
    if os.path.exists(ave_output_file) and os.path.exists(lstm_output_file):
        return

    model_path = os.path.join(global_conf['model']['word_elmo'], treebank_conf['elmo_model'])
    cmds = global_conf['exec']['elmo'] + ['test', '--model', model_path, '--input_format', 'conll',
                                          '--input', input_file, '--output_ave', ave_output_file,
                                          '--output_lstm', lstm_output_file]

    print('running parser aux commands for {}'.format(treebank_conf['code']), file=sys.stderr)
    print(' '.join(cmds), file=sys.stderr)
    pipe = subprocess.Popen(cmds)
    pipe.wait()


def _parse_stanford(treebank_conf, global_conf, input_file, output_file):
    """

    :param treebank_conf:
    :param global_conf:
    :param input_file:
    :param output_file:
    :return:
    """

    range_match = re.search('\[(\d)-(\d)\]', treebank_conf['parse_model'])

    need_exchange45 = False
    final_output_file = output_file
    if range_match is not None:
        model_payloads = []
        for i in range(int(range_match.group(1)), int(range_match.group(2)) + 1):
            model_payloads.append(os.path.join(global_conf['model']['parse_model_dir'],
                                               re.sub('\[\d-\d\]', str(i), treebank_conf['parse_model'])))
        print(range_match.group())
        if '_transfer' in model_payloads[0]:
            need_exchange45 = True
            output_file = output_file + '.45'

        cmds = global_conf['exec']['stanford'] + ['--save_dir', model_payloads[0], 'ensemble', input_file,
                                                  '--output_file', output_file,
                                                  '--sum_type', 'prob',
                                                  '--other_save_dirs'] + model_payloads[1:]
    else:
        model_path = os.path.join(global_conf['model']['parse_model_dir'], treebank_conf['parse_model'])
        if '_transfer' in model_path:
            need_exchange45 = True
            output_file = output_file + '.45'

        cmds = global_conf['exec']['stanford'] + ['--save_dir', model_path, 'parse', input_file,
                                                  '--output_file', output_file]

    if treebank_conf['embeddings'] != '':
        cmds.append('--pretrained_vocab')
        cmds.append('filename={0}'.format(os.path.join(global_conf['resources']['word_embeddings'],
                                                       treebank_conf['embeddings'])))

    model_path = os.path.join(global_conf['model']['parse_model_dir'], treebank_conf['parse_model'])
    _, model_dir = os.path.split(model_path)
    elmo_type = model_dir.split('-')[1]
    if elmo_type == 'lstm':
        cmds.append('--elmo_file')
        cmds.append(os.path.join(global_conf['output'], 'words.lstm_elmo'))
    elif elmo_type == 'ave':
        cmds.append('--elmo_file')
        cmds.append(os.path.join(global_conf['output'], 'words.ave_elmo'))

    print('running parse commands for {}'.format(treebank_conf['code']), file=sys.stderr)
    print(' '.join(cmds), file=sys.stderr)
    pipe = subprocess.Popen(cmds)
    pipe.wait()

    if need_exchange45:
        cmds = global_conf['exec']['exchange45'] + [output_file, final_output_file]
        print('running exchange45 commands for {}'.format(treebank_conf['code']), file=sys.stderr)
        print(' '.join(cmds), file=sys.stderr)
        pipe = subprocess.Popen(cmds)
        pipe.wait()
        os.remove(output_file)


def _parse_trans_parser(treebank_conf, global_conf, input_file, output_file):
    """

    :param treebank_conf:
    :param global_conf:
    :param input_file:
    :param output_file:
    :return:
    """
    model_dir = os.path.join(global_conf['model']['parse_model_dir'], treebank_conf['parse_model'])
    info_file = os.path.join(model_dir, '{0}.infos'.format(treebank_conf['code']))
    model1_file = os.path.join(model_dir, [f for f in os.listdir(model_dir) if '-v0' and '.params' in f][0])
    model2_file = os.path.join(model_dir, [f for f in os.listdir(model_dir) if '-v1' and '.params' in f][0])

    cmds = global_conf['exec']['trans_parser'] + ['--dynet_seed', '1234', '--dynet_mem', '4000', '-p', input_file,
                                                  '-i', info_file, '-m', model1_file, '--model2', model2_file,
                                                  '-D', '-s', 'list-tree',
                                                  '-k', '{0}-all-v3'.format(treebank_conf['code']),
                                                  '--pretrained_dim', '100', '--hidden_dim', '100',
                                                  '--bilstm_hidden_dim', '200', '--lstm_input_dim', '200',
                                                  '--input_dim', '100', '--action_dim', '50', '--pos_dim', '50',
                                                  '--rel_dim', '50', '-P', '-B', '-R']

    fpo = open(output_file, 'w')
    print('running parse commands for {}'.format(treebank_conf['code']), file=sys.stderr)
    print(' '.join(cmds), file=sys.stderr)
    pipe = subprocess.Popen(cmds, stdout=fpo)
    pipe.wait()


def parse(treebank_conf, global_conf, output_dir, info):
    """

    :param treebank_conf:
    :param global_conf:
    :param output_dir:
    :param info:
    :return:
    """
    input_file = os.path.join(global_conf['output'], 'tagged.conllu')
    output_file = os.path.join(output_dir, info['outfile'])

    try:
        if treebank_conf['parser'] == 'stanford':
            _parse_stanford(treebank_conf, global_conf, input_file, output_file)
        elif treebank_conf['parser'] == 'trans_parser':
            _parse_trans_parser(treebank_conf, global_conf, input_file, output_file)
        else:
            raise ValueError('Unknown parser {0}'.format(treebank_conf['parser']))
    except Exception:
        print('Parse failed, downgrade to copy', file=sys.stderr)
        shutil.copy(input_file, output_file)


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
    cmd.add_argument('-output_dir', required=True, help='the path to the output dir')
    opts = cmd.parse_args()

    # get global configuration
    global_conf = load_global_conf()

    global_conf['output'] = os.path.join(opts.output_dir, 'tmp')
    if not os.path.exists(global_conf['output']):
        os.makedirs(global_conf['output'])

    dataset_conf = load_conf()

    meta = json.load(open(os.path.join(opts.input_dir, 'metadata.json')))
    for info in sorted(meta, key=lambda entry: (entry['lcode'] in ('zh', 'ja', 'vi'), entry['lcode'], entry['tcode']),
                       reverse=True):
        for f in glob.glob(os.path.join(global_conf['output'], '*')):
            os.remove(f)

        code = '{0}_{1}'.format(info['lcode'], info['tcode'])
        treebank_conf = dataset_conf[code]

        # sentence segmentation
        sentence_segment(treebank_conf, global_conf, opts.input_dir, info)

        # tokenize aux
        tokenize_aux(treebank_conf, global_conf)

        # tokenize
        tokenize(treebank_conf, global_conf)

        # morphology_tag
        morphology_tag(treebank_conf, global_conf)

        # postag aux
        postag_aux(treebank_conf, global_conf)

        # postag
        postag(treebank_conf, global_conf)

        # parse aux
        parse_aux(treebank_conf, global_conf)

        # parse
        parse(treebank_conf, global_conf, opts.output_dir, info)


if __name__ == "__main__":
    main()
