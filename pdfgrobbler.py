#!/usr/bin/env python
#
# Copyright 2016 Ben Walsh
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import numpy as np
import logging
import datetime
import argparse
import csv

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter, PDFConverter
from pdfminer.layout import LAParams, LTContainer, LTPage, LTText, LTLine, LTRect, LTCurve, LTFigure, LTImage, LTChar, LTTextLine, LTTextBox, LTTextBoxHorizontal, LTTextBoxVertical, LTTextGroup, LTAnno, LTTextLineHorizontal
from pdfminer.pdfpage import PDFPage
from cStringIO import StringIO

_logger = logging.getLogger(__name__)


def _get_text_xs(item, prev_x1=0.0):
    if isinstance(item, LTContainer):
        res = []
        x1 = 0.0
        for child in item:
            child_xs = _get_text_xs(child, x1)
            x1 = child_xs[-1]
            res.extend(child_xs[:-1])
        res.append(x1)
        return res
    elif isinstance(item, LTAnno):
        assert len(item.get_text()) == 1
        return [prev_x1, prev_x1]
    elif isinstance(item, LTText):
        assert len(item.get_text()) == 1
        return [item.x0, item.x1]
    assert False, 'Unknown type'


class TextFinder(PDFConverter):
    def __init__(self, rsrcmgr, outfp, codec='utf-8', pageno=1, laparams=None,
                 showpageno=False, imagewriter=None):
        PDFConverter.__init__(self, rsrcmgr, outfp, codec=codec, pageno=pageno, laparams=laparams)
        self.showpageno = showpageno
        self.imagewriter = imagewriter
        self.texts = []


    def receive_layout(self, ltpage):
        def render(item):
            if isinstance(item, LTTextLineHorizontal):
                self.texts.append(item)
            if isinstance(item, LTContainer):
                for child in item:
                    render(child)
            elif isinstance(item, LTImage):
                if self.imagewriter is not None:
                    self.imagewriter.export_image(item)
        render(ltpage)
        return

    # Some dummy functions to save memory/CPU when all that is wanted
    # is text.  This stops all the image and drawing ouput from being
    # recorded and taking up RAM.
    def render_image(self, name, stream):
        if self.imagewriter is None:
            return
        PDFConverter.render_image(self, name, stream)
        return

    def paint_path(self, gstate, stroke, fill, evenodd, path):
        return


def _samerow(e0, e1):
    # XXX
    e0h = (e0.y1 - e0.y0) / 3.0
    e1h = (e1.y1 - e1.y0) / 3.0
    return (e0.y0 < e1.y0 + e1h and e0.y0 + e0h > e1.y0)


def _elemcmp(e0, e1):
    if _samerow(e0, e1):
        return cmp(e0.x0, e1.x0)
    return -cmp(e0.y0, e1.y0)  # higher y first (top of page first)


def _cleanstr(s):
    return s.strip().replace('\n', ' ')


def clean_item_text(item):
    return _cleanstr(item.get_text())


def _find_in_row(row, txt, x0=0.0):
    if x0 is None:
        return (None, None)
    for j in xrange(len(row)):
        text = row[j].get_text().lower()
        index = text.find(txt)
        if index < 0:
            continue
        xs = _get_text_xs(row[j])  # get x0 for every char
        assert len(xs) == len(text) + 1
        if xs[index] > x0:
            return (xs[index], xs[index + len(txt)])
    return (None, None)


def find_header_row(rows, headers):
    for i in xrange(len(rows)):
        row = rows[i]

        header_xs = []
        prev_x1 = 0.0
        for header in headers:
            x0, x1 = _find_in_row(row, header, prev_x1)
            if x0 is None:
                break
            approx_char_width = (x1 - x0) / float(len(header))
            header_xs.append(x0 - approx_char_width)
            prev_x1 = x1

        if len(header_xs) == len(headers):
            return tuple([i] + header_xs)

    return tuple([None] + ([None] * len(headers)))


def split_row(row, colxs):
    colj = 0
    ncols = len(colxs)
    vals = [[] for colx in colxs]
    for item in row:
        xs = _get_text_xs(item)
        text = item.get_text()
        assert len(xs) == len(text) + 1
        for (c, x) in zip(text, xs):
            if x < 0.0:  # non-printing char
                vals[colj].append(' ')
                continue
            while colj > 0 and x < colxs[colj]:
                colj -= 1
            while colj < (ncols - 1) and x >= colxs[colj + 1]:
                colj += 1
            vals[colj].append(c)

    return [_cleanstr(''.join(v)) for v in vals]


def _elems_to_rows(texts):
    rows = []
    row = []
    for i in xrange(len(texts)):
        if i > 0 and (texts[i].x0 < texts[i - 1].x0 or not _samerow(texts[i - 1], texts[i])):
            rows.append(row)
            row = []
        row.append(texts[i])
    rows.append(row)

    return rows


def parse_float(s):
    chars = []
    for c in s:
        if not c.isdigit() and c not in ('.', '-'):
            continue
        chars.append(c)
    return float(''.join(chars))


def convert_pdf_to_txt(path, func):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextFinder(rsrcmgr, retstr, codec=codec, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos=set()

    with open(path, 'rb') as fp:
        for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching, check_extractable=True):
            device.texts = []
            interpreter.process_page(page)
            page_texts = device.texts[:]
            page_texts.sort(_elemcmp)  # (0, 0) == bottom-left
            rows = _elems_to_rows(page_texts)
            func(rows)

    device.close()
    retstr.close()
