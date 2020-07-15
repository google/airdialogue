# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Python implementation of BLEU and smooth-BLEU.

This module provides a Python implementation of BLEU and smooth-BLEU.
Smooth BLEU is computed following the method outlined in the paper:
Chin-Yew Lin, Franz Josef Och. ORANGE: a method for evaluating automatic
evaluation metrics for machine translation. COLING 2004.
"""

import collections
import math
import tqdm

from multiprocessing import Pool
from functools import partial


def _get_ngrams(segments, order):
  """Extracts all n-grams upto a given maximum order from an input segment.

  Args:
    segments: text segment from which n-grams will be extracted.
    order: n of the n-grams returned.

  Returns:
    The Counter containing all n-grams in segment
    with a count of how many times each n-gram occurred.
  """
  ngram_counts = collections.Counter()
  for seg in segments:
    for i in range(0, len(seg) - order + 1):
      ngram = tuple(seg[i:i+order])
      ngram_counts[ngram] += 1
  return ngram_counts


def compute_kl(reference_corpus,
               translation_corpus,
               max_order=4,
               freq_thre=100,
               workers=20,
               verbose=False):
  """Computes KLdivergence of translated segments against one or more references.

  Args:
    reference_corpus: list of lists of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    smooth: Whether or not to apply Lin et al. 2004 smoothing.

  Returns:
    3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
    precisions and brevity penalty.
  """
  results = {}
  itertool = tqdm.tqdm if verbose else lambda x, *args, **kw: x

  for order in range(1, max_order + 1):
    with Pool(processes=workers) as pool:
      chunksize = len(reference_corpus)//(3*workers)
      chunkedexamples = [reference_corpus[i:i + chunksize]
                         for i in range(0, len(reference_corpus), chunksize)]
      ngram_counts = list(itertool(pool.imap(partial(_get_ngrams, order=order), chunkedexamples),\
                                    total=len(chunkedexamples), \
                                    desc="Calculating {}-gram for ref data".format(order)))
      merged_ref_ngram_counts = collections.Counter()
      for c in ngram_counts:
        merged_ref_ngram_counts += c
    with Pool(processes=workers) as pool:
      chunksize = len(translation_corpus)//(10*workers)
      chunkedexamples = [translation_corpus[i:i + chunksize]
                         for i in range(0, len(translation_corpus), chunksize)]
      ngram_counts = list(itertool(pool.imap(partial(_get_ngrams, order=order), chunkedexamples),\
                                    total=len(chunkedexamples), \
                                    desc="Calculating {}-gram for trans data".format(order)))
      merged_trans_ngram_counts = collections.Counter()
      for c in ngram_counts:
        merged_trans_ngram_counts += c

    merged_ref_ngram_counts = \
      collections.Counter({k: c for k, c in merged_ref_ngram_counts.items() if c >= freq_thre})
    ref_total_nums = sum([c for k, c in merged_ref_ngram_counts.items()])
    # We do not remove items from merged_trans_ngram_counts to prevent infite issue
    trans_total_nums = sum([c for k, c in merged_trans_ngram_counts.items() if c >= freq_thre])

    kl = 0
    # eps: smoothing parameter
    eps = 1e-10
    for k, c in merged_ref_ngram_counts.items():
      _c = merged_trans_ngram_counts.get(k, 0)

      # apply smoothing
      c += eps
      _c += eps

      kl += - c/ref_total_nums * math.log((_c/c)*(ref_total_nums/trans_total_nums))

    results["{}-gram".format(order)] = kl

  return results
