#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import warnings
warnings.filterwarnings("ignore")

INDIC_NLP_LIB_HOME = 'indic_nlp_library'
INDIC_NLP_RESOURCES = 'indic_nlp_resources'

sys.path.append('{}/src'.format(INDIC_NLP_LIB_HOME))

from indicnlp import common, loader
common.set_resources_path(INDIC_NLP_RESOURCES)
loader.load()

from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize import indic_normalize
from indicnlp.tokenize import sentence_tokenize

normalizer_factory = indic_normalize.IndicNormalizerFactory()


def preprocess_sent(text, lang):
    """
    Pre-process text (normalization and tokenization).

    text: text string to preprocess
    lang: language code (2-letter ISO code)

    returns the processed text string
    """
    normalizer = normalizer_factory.get_normalizer(lang)

    return indic_tokenize.trivial_tokenize(normalizer.normalize(
                                           text.replace('\n', ' ')), lang)


def sent_split(text, lang):
    """
    Sentence splitter.

    text: text to sentence split
    lang: language

    returns list of sentences
    """
    return sentence_tokenize.sentence_split(text, lang)
