# Stage 6C Hugging Face Dataset Inspection Report

## Overview
- **Dataset Name:** `jon-tow/starcoderdata-python-edu`
- **Dataset Columns:** ['max_stars_repo_path', 'max_stars_repo_name', 'max_stars_count', 'id', 'content', 'score', 'int_score']

## Exact Filtering Rules Used
1. Must be parseable by `ast.parse` without SyntaxError.
2. Must contain exactly 1 function definition (`ast.FunctionDef` or `ast.AsyncFunctionDef`).
3. Must be <= 2000 characters in length.
4. Must not contain top-level function execution calls.
5. Decontamination Rules: Must NOT contain any of the following banned keywords: `HumanEval`, `openai_humaneval`, `canonical_solution`, `check(candidate)`, `METADATA`, `from humaneval`, `"__main__"`, `doctest.testmod`, `unittest.main`, ````python`, `eval(`, `exec(`, `subprocess`, `os.system`.

## Contamination/Decontamination Findings
- Tested against HumanEval metadata and testing boilerplate.
- The raw dataset contains many modules, classes, and unparseable scripts. 
- After applying the strict rules, any example that could be a test execution or standard completion dataset artifact was successfully filtered out.

## Statistics (from 1000 streamed rows)
- **Parse Rate:** 48.4%
- **Function-Definition Rate:** 37.5%
- **Suitability for prompt -> full-function conversion:** 3.6% met all exact filtering rules.

## 20 Raw Examples

### Raw 1
```python
<reponame>MTES-MCT/sparte
from rest_framework_gis import serializers
from rest_framework import serializers as s

from .models import (
    Artificialisee2015to2018,
    Artificielle2018,
    CommunesSybarval,
    CouvertureSol,
    EnveloppeUrbaine2018,
    Ocsge,
    Renaturee2018to2015,
    Sybarval,
    Voirie2018,
    ZonesBaties2018,
    UsageSol,
)


def get_label(code="", label=""):
    if code is None:
        code = "-"
    if label is None:
        label = "inconnu"
    return f"{code...
```

### Raw 2
```python
from django.contrib import admin
from .models import SearchResult

# Register your models here.
class SearchResultAdmin(admin.ModelAdmin):
    fields = ["query", "heading", "url", "text"]

admin.site.register(SearchResult, SearchResultAdmin)...
```

### Raw 3
```python
import asyncio
import os
import tempfile
from contextlib import ExitStack
from typing import Text, Optional, List, Union, Dict

from rasa.importers.importer import TrainingDataImporter
from rasa import model
from rasa.model import FingerprintComparisonResult
from rasa.core.domain import Domain
from rasa.utils.common import TempDirectoryPath

from rasa.cli.utils import (
    print_success,
    print_warning,
    print_error,
    bcolors,
    print_color,
)
from rasa.constants import DEFAULT_MODEL...
```

### Raw 4
```python
<gh_stars>1-10
class Solution:
    def finalPrices(self, prices: List[int]) -> List[int]:
        res = []
        for i in range(len(prices)):
            for j in range(i+1,len(prices)):
                if prices[j]<=prices[i]:
                    res.append(prices[i]-prices[j])
                    break
                if j==len(prices)-1:
                    res.append(prices[i])
        res.append(prices[-1])
        return res...
```

### Raw 5
```python
<gh_stars>0
# ============================================================================
# FILE: default.py
# AUTHOR: <NAME> <<EMAIL> at g<EMAIL>>
# License: MIT license
# ============================================================================

import re
import typing

from denite.util import echo, error, clearmatch, regex_convert_py_vim
from denite.util import Nvim, UserContext, Candidates, Candidate
from denite.parent import SyncParent


class Default(object):
    @property
    def is_a...
```

### Raw 6
```python
<filename>PyDSTool/core/context_managers.py
# -*- coding: utf-8 -*-

"""Context managers implemented for (mostly) internal use"""

import contextlib
import functools
from io import UnsupportedOperation
import os
import sys


__all__ = ["RedirectStdout", "RedirectStderr"]


@contextlib.contextmanager
def _stdchannel_redirected(stdchannel, dest_filename, mode="w"):
    """
    A context manager to temporarily redirect stdout or stderr

    Originally by <NAME>, 2013
    (http://marc-abramowitz.com...
```

### Raw 7
```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from . import __version__ as app_version

app_name = "pos_kiosk"
app_title = "Pos Kiosk"
app_publisher = "9t9it"
app_description = "Kiosk App"
app_icon = "octicon octicon-file-directory"
app_color = "grey"
app_email = "<EMAIL>"
app_license = "MIT"

# Includes in <head>
# ------------------

# include js, css files in header of desk.html
# app_include_css = "/assets/pos_kiosk/css/pos_kiosk.css"
# app_include_js = "/assets/pos_kiosk/j...
```

### Raw 8
```python
<gh_stars>1-10
from keras import Model, Input
from keras.layers import Dense, concatenate, LSTM, Reshape, Permute, Embedding, Dropout, Convolution1D, Flatten
from keras.optimizers import Adam

from pypagai.models.base import KerasModel


class SimpleLSTM(KerasModel):
    """
    Use a simple lstm neural network
    """
    @staticmethod
    def default_config():
        config = KerasModel.default_config()
        config['hidden'] = 32

        return config

    def __init__(self, cfg):
       ...
```

### Raw 9
```python
<filename>lib/variables/latent_variables/__init__.py
from .fully_connected import FullyConnectedLatentVariable
from .convolutional import ConvolutionalLatentVariable...
```

### Raw 10
```python
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:
''' PNASNet in PyTorch.
Paper: Progressive Neural Architecture Search
'''

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.backbone_name import BackboneName
from easyai.model.backbone.utility.base_backbone import *
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.cls.pnasnet_block import CellA, CellB

__all__ = ['pnasnet_A', 'pnasnet_B...
```

### Raw 11
```python
# -*- coding: utf-8 -*-
#  coding=utf-8
import json
import os
import math
import logging
import requests
import time

from map_download.cmd.BaseDownloader import DownloadEngine, BaseDownloaderThread, latlng2tile_terrain, BoundBox


def get_access_token(token):
    resp = None
    request_count = 0
    url = "https://api.cesium.com/v1/assets/1/endpoint"
    while True:
        if request_count > 4:
            break
        try:
            request_count += 1
            param = {'access_token': ...
```

### Raw 12
```python
<reponame>vahini01/electoral_rolls
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 23:28:58 2017

@author: dhingratul
"""
import urllib.request
import os
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from bs4 import BeautifulSoup
import ssl
import requests
import wget
from PyPDF2 import PdfFileReader


def download_file(pdf_url, mdir, filename, flag=False):
    if flag is True:
        context = ssl._create_unverified_context()
        r...
```

### Raw 13
```python
<gh_stars>0
"""
Experiment summary
------------------
Treat each province/state in a country cases over time
as a vector, do a simple K-Nearest Neighbor between
countries. What country has the most similar trajectory
to a given country?

Plots similar countries
"""

import sys
sys.path.insert(0, '..')

from utils import data
import os
import sklearn
import numpy as np
import json
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# ------------ HYPERPARAMETERS -------------
BASE_...
```

### Raw 14
```python
<reponame>steven-lang/rational_activations
"""
Rational Activation Functions for MXNET
=======================================

This module allows you to create Rational Neural Networks using Learnable
Rational activation functions with MXNET networks.
"""
import mxnet as mx
from mxnet import initializer
from mxnet.gluon import HybridBlock

from rational.utils.get_weights import get_parameters
from rational.mxnet.versions import _version_a, _version_b, _version_c, _version_d
from rational._base....
```

### Raw 15
```python
<filename>torchflare/criterion/utils.py<gh_stars>1-10
"""Utils for criterion."""
import torch
import torch.nn.functional as F


def normalize(x, axis=-1):
    """Performs L2-Norm."""
    num = x
    denom = torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12
    return num / denom


# Source : https://github.com/earhian/Humpback-Whale-Identification-1st-/blob/master/models/triplet_loss.py
def euclidean_dist(x, y):
    """Computes Euclidean distance."""
    m, n = x.size(0), y.size(0)
    x...
```

### Raw 16
```python
"""Tests for the sbahn_munich integration"""


line_dict = {
    "name": "S3",
    "color": "#333333",
    "text_color": "#444444",
}...
```

### Raw 17
```python
<reponame>geudrik/hautomation
#! /usr/bin/env python2.7
# -*- coding: latin-1 -*-

from flask import Blueprint
from flask import current_app
from flask import render_template

from flask_login import login_required

homestack = Blueprint("homestack", __name__, url_prefix="/homestack")


@homestack.route("/", methods=["GET"])
@login_required
def home():
    return render_template("homestack/home.html")...
```

### Raw 18
```python
"""Forms for RTD donations"""

import logging

from django import forms
from django.conf import settings
from django.utils.translation import ugettext_lazy as _

from readthedocs.payments.forms import StripeModelForm, StripeResourceMixin
from readthedocs.payments.utils import stripe

from .models import Supporter

log = logging.getLogger(__name__)


class SupporterForm(StripeResourceMixin, StripeModelForm):

    """Donation support sign up form

    This extends the basic payment form, giving fi...
```

### Raw 19
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .base import DataReaderBase
from ..tools import COL, _get_dates, to_float, to_int

import pandas as pd
#from pandas.tseries.frequencies import to_offset
from six.moves import cStringIO as StringIO
import logging
import traceback
import datetime

import json
import token, tokenize


def ymd_to_date(y, m, d):
    """
    Returns date

    >>> expiration = {u'd': 1, u'm': 12, u'y': 2014}
    >>> ymd_to_date(**expiration)
    datetime.date(2014, 12...
```

### Raw 20
```python
<reponame>Vail-qin/Keras-TextClassification
# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/11/2 21:08
# @author  : Mo
# @function:


from keras_textclassification.data_preprocess.text_preprocess import load_json, save_json
from keras_textclassification.conf.path_config import path_model_dir
path_fast_text_model_vocab2index = path_model_dir + 'vocab2index.json'
path_fast_text_model_l2i_i2l = path_model_dir + 'l2i_i2l.json'

import numpy as np
import os


class PreprocessGenerator:
 ...
```

## 30 Accepted Examples After Filtering

### Accepted 1
```python
from django.db.models import Q

from django.shortcuts import render
from django.http import Http404

# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view

from .models import Product, Category
from .serializers import ProductSerializer, CategorySerializer

class LatestProductsList(APIView):
    def get(self, request, format=None):
        products = Product.objects.all()[0:4]
        serializer = ProductSerializer(products,many=True)
        return Response(serializer.data)

class ProductDetail(APIView):
    def get_object(self, category_slug, product_slug):
        try:
            return Product.objects.filter(category__slug=category_slug).get(slug=product_slug)
        except Product.DoesNotExist:
            raise Http404

    def get(self, request, category_slug, product_slug, format= None):
        product = self.get_object(category_slug, product_slug)
        serializer = ProductSerializer(product)
        return Response(serializer.data)

class CategoryDetail(APIView):
    def get_object(self, category_slug):
        try:
            return Category.objects.get(slug=category_slug)
        except Category.DoesNotExist:
            raise Http404
    
    def get(self, request, category_slug, format= None):
        category = self.get_object(category_slug)
        serializer = CategorySerializer(category)
        return Response(serializer.data)

@api_view(['POST'])
def search(request):
    query = request.data.get('query', '')

    if query:
        products = Product.objects.filter(Q(name__icontains=query) | Q(description__icontains=query))
        serializer = ProductSerializer(products, many=True)
        return Response(serializer.data)
    else:
        return Response({"products": []})
```

### Accepted 2
```python
__all__ = ('create_partial_webhook_from_id', )

from scarletio import export

from ..core import USERS

from .preinstanced import WebhookType
from .webhook import Webhook


@export
def create_partial_webhook_from_id(webhook_id, token, *, type_=WebhookType.bot, channel_id=0):
    """
    Creates a partial webhook from the given parameters. If the webhook with the given `webhook_id` already exists,
    then returns that instead.
    
    Parameters
    ----------
    webhook_id : `int`
        The identifier number of the webhook.
    token : `str`
        The token of the webhook.
    type_ : ``WebhookType`` = `WebhookType.bot`, Optional (Keyword only)
        The webhook's type. Defaults to `WebhookType.bot`.
    channel_id : `int` = `0`, Optional (Keyword only)
        The webhook's channel's identifier. Defaults to `0`.
    
    Returns
    -------
    webhook : ``Webhook``
    """
    try:
        webhook = USERS[webhook_id]
    except KeyError:
        webhook = Webhook._create_empty(webhook_id)
        webhook.channel_id = channel_id
        webhook.type = type_
        
        USERS[webhook_id] = webhook
    
    webhook.token = token
    return webhook
```

### Accepted 3
```python
#!/usr/bin/env python

# runs after the job (and after the default post-filter)
from galaxy.tools.parameters import DataToolParameter
# Older py compatibility
try:
    set()
except:
    from sets import Set as set


def validate_input( trans, error_map, param_values, page_param_map ):
    dbkeys = set()
    data_param_names = set()
    data_params = 0
    for name, param in page_param_map.items():
        if isinstance( param, DataToolParameter ):
            # for each dataset parameter
            if param_values.get(name, None) is not None:
                dbkeys.add( param_values[name].dbkey )
                data_params += 1
                # check meta data
                try:
                    param = param_values[name]
                    int( param.metadata.startCol )
                    int( param.metadata.endCol )
                    int( param.metadata.chromCol )
                    if param.metadata.strandCol is not None:
                        int( param.metadata.strandCol )
                except:
                    error_msg = ("The attributes of this dataset are not properly set. "
                        "Click the pencil icon in the history item to set the chrom, start, end and strand columns.")
                    error_map[name] = error_msg
            data_param_names.add( name )
    if len( dbkeys ) > 1:
        for name in data_param_names:
            error_map[name] = "All datasets must belong to same genomic build, " \
                "this dataset is linked to build '%s'" % param_values[name].dbkey
    if data_params != len(data_param_names):
        for name in data_param_names:
            error_map[name] = "A dataset of the appropriate type is required"
```

### Accepted 4
```python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Config(object):
    pass


def read_params():
    cfg = Config()

    #params for text detector
    cfg.det_algorithm = "DB"
    cfg.det_model_dir = "./inference/ch_ppocr_mobile_v2.0_det_infer/"
    cfg.det_limit_side_len = 960
    cfg.det_limit_type = 'max'

    #DB parmas
    cfg.det_db_thresh = 0.3
    cfg.det_db_box_thresh = 0.5
    cfg.det_db_unclip_ratio = 1.6
    cfg.use_dilation = False

    # #EAST parmas
    # cfg.det_east_score_thresh = 0.8
    # cfg.det_east_cover_thresh = 0.1
    # cfg.det_east_nms_thresh = 0.2

    cfg.use_pdserving = False
    cfg.use_tensorrt = False

    return cfg
```

### Accepted 5
```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from .models import User
from .twitter import vectorize_tweet


def predict_user(user1_name, user2_name, tweet_text):
    """
    Determine and return which user is more likely to say a given Tweet.
    
    Example: predict_user('ausen', 'elonmusk', 'Lambda School Rocks!')
    Returns 1 corresponding to 1st user passed in, or 0 for second.
    """
    user1 = User.query.filter(User.name == user1_name).one()
    user2 = User.query.filter(User.name == user2_name).one()
    user1_vect = np.array([tweet.vect for tweet in user1.tweets])
    user2_vect = np.array([tweet.vect for tweet in user2.tweets])

    vects = np.vstack([user1_vect, user2_vect])
    labels = np.concatenate([np.ones(len(user1.tweets)), 
                             np.zeros(len(user2.tweets))])
    log_reg = LogisticRegression().fit(vects, labels)
    # We've done the model fitting, now to predict...
    hypo_tweet_vect = vectorize_tweet(tweet_text)
    return log_reg.predict(np.array(hypo_tweet_vect).reshape(1,-1))
```

### Accepted 6
```python
import discord
import random
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import csv


async def plot_user_activity(client, ctx):
    plt.style.use('fivethirtyeight')
    df = pd.read_csv('innovators.csv', encoding= 'unicode_escape')

    author = df['author'].to_list()

    message_counter = {}

    for i in author:
        if i in message_counter:
            message_counter[i] += 1
        else:
            message_counter[i] = 1
    
    # for not mentioning the bot in the line graph. 
    message_counter.pop('ninza_bot_test')
    
    authors_in_discord = list(message_counter.keys())
    no_of_messages = list(message_counter.values())

    plt.plot(authors_in_discord, no_of_messages, marker = 'o', markersize=10)
    plt.title('msg sent by author in the server.')
    plt.xlabel('Author')
    plt.ylabel('Message_count')

    plt.savefig('output2.png')
    plt.tight_layout()
    plt.close()

    await ctx.send(file = discord.File('output2.png'))
```

### Accepted 7
```python
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the rotLeft function below.
def rotLeft(a, d):
    alist = list(a)
    b = alist[d:]+alist[:d]
    return b

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nd = input().split()

    n = int(nd[0])

    d = int(nd[1])

    a = list(map(int, input().rstrip().split()))

    result = rotLeft(a, d)

    fptr.write(' '.join(map(str, result)))
    fptr.write('\n')

    fptr.close()
```

### Accepted 8
```python
def indexof(listofnames, value):
    if value in listofnames:
        value_index = listofnames.index(value)
        return(listofnames, value_index)
    else: return(-1)
```

### Accepted 9
```python
#coding:utf-8
#
# id:           functional.index.create.03
# title:        CREATE ASC INDEX
# decription:   CREATE ASC INDEX
#               
#               Dependencies:
#               CREATE DATABASE
#               CREATE TABLE
#               SHOW INDEX
# tracker_id:   
# min_versions: []
# versions:     1.0
# qmid:         functional.index.create.create_index_03

import pytest
from firebird.qa import db_factory, isql_act, Action

# version: 1.0
# resources: None

substitutions_1 = []

init_script_1 = """CREATE TABLE t( a INTEGER);
commit;"""

db_1 = db_factory(sql_dialect=3, init=init_script_1)

test_script_1 = """CREATE ASC INDEX test ON t(a);
SHOW INDEX test;"""

act_1 = isql_act('db_1', test_script_1, substitutions=substitutions_1)

expected_stdout_1 = """TEST INDEX ON T(A)"""

@pytest.mark.version('>=1.0')
def test_1(act_1: Action):
    act_1.expected_stdout = expected_stdout_1
    act_1.execute()
    assert act_1.clean_expected_stdout == act_1.clean_stdout
```

### Accepted 10
```python
from tools.geofunc import GeoFunc
import pandas as pd
import json

def getData(index):
    '''报错数据集有（空心）：han,jakobs1,jakobs2 '''
    '''形状过多暂时未处理：shapes、shirt、swim、trousers'''
    name=["ga","albano","blaz1","blaz2","dighe1","dighe2","fu","han","jakobs1","jakobs2","mao","marques","shapes","shirts","swim","trousers"]
    print("开始处理",name[index],"数据集")
    '''暂时没有考虑宽度，全部缩放来表示'''
    scale=[100,0.5,100,100,20,20,20,10,20,20,0.5,20,50]
    print("缩放",scale[index],"倍")
    df = pd.read_csv("data/"+name[index]+".csv")
    polygons=[]
    for i in range(0,df.shape[0]):
        for j in range(0,df['num'][i]):
            poly=json.loads(df['polygon'][i])
            GeoFunc.normData(poly,scale[index])
            polygons.append(poly)
    return polygons
```

### Accepted 11
```python
# This file is part of Indico.
# Copyright (C) 2002 - 2020 CERN
#
# Indico is free software; you can redistribute it and/or
# modify it under the terms of the MIT License; see the
# LICENSE file for more details.

from flask import redirect

from indico.modules.events.abstracts.models.abstracts import Abstract
from indico.web.flask.util import url_for
from indico.web.rh import RHSimple


@RHSimple.wrap_function
def compat_abstract(endpoint, confId, friendly_id, track_id=None, management=False):
    abstract = Abstract.find(event_id=confId, friendly_id=friendly_id).first_or_404()
    return redirect(url_for('abstracts.' + endpoint, abstract, management=management))
```

### Accepted 12
```python
import matplotlib.pyplot
__author__ = 'xiongyi'
line1 = [(200, 100), (200, 400)]
line2 = [(190, 190), (210, 210)]
def overlap():
    l1p1x = line1[0][0]
    l1p1y = line1[0][1]
    l1p2x = line1[1][0]
    l1p2y = line1[1][1]
    # make sure p1x < p2x
    if l1p1x > l1p2x:
        tmp = l1p1x
        l1p1x = l1p2x
        l1p2x = tmp
    # make sure p1y < p2y
    if l1p1y > l1p2y:
        tmp = l1p1y
        l1p1y = l1p2y
        l1p2y = tmp
    l2p1x = line2[0][0]
    l2p1y = line2[0][1]
    l2p2x = line2[1][0]
    l2p2y = line2[1][1]
    # make sure p1x < p2x
    if l2p1x > l2p2x:
        tmp = l2p1x
        l2p1x = l2p2x
        l2p2x = tmp
    # make sure p1y < p2y
    if l2p1y > l2p2y:
        tmp = l2p1y
        l2p1y = l2p2y
        l2p2y = tmp

    # line2 rectangle is inside line1 rect
    if l1p1x < l2p2x and l1p2x > l2p1x and l1p1y < l2p2y and l1p2y > l2p1y:
        return True
    # line2 rectangle is inside line1 rect
    if l1p1x > l2p2x and l1p2x < l2p1x and l1p1y > l2p2y and l1p2y < l2p1y:
        return True
    if l1p1x > l2p2x or l1p2x < l2p1x:
        return False
    if l1p1y > l2p2y or l1p2y < l2p1y:
        return False
    return True

if __name__ == '__main__':
    matplotlib.pyplot.plot((line1[0][0],line1[1][0]),(line1[0][1],line1[1][1]))
    matplotlib.pyplot.hold(True)

    matplotlib.pyplot.plot((line2[0][0],line2[1][0]),(line2[0][1],line2[1][1]))
    print(overlap())
    matplotlib.pyplot.show()
```

### Accepted 13
```python
# This file is part of Patsy
# Copyright (C) 2013 <NAME> <<EMAIL>>
# See file LICENSE.txt for license information.

# Regression tests for fixed bugs (when not otherwise better covered somewhere
# else)

from patsy import (EvalEnvironment, dmatrix, build_design_matrices,
                   PatsyError, Origin)

def test_issue_11():
    # Give a sensible error message for level mismatches
    # (At some points we've failed to put an origin= on these errors)
    env = EvalEnvironment.capture()
    data = {"X" : [0,1,2,3], "Y" : [1,2,3,4]}
    formula = "C(X) + Y"
    new_data = {"X" : [0,0,1,2,3,3,4], "Y" : [1,2,3,4,5,6,7]}
    info = dmatrix(formula, data)
    try:
        build_design_matrices([info.design_info], new_data)
    except PatsyError as e:
        assert e.origin == Origin(formula, 0, 4)
    else:
        assert False
```

### Accepted 14
```python
__all__ = ["load"]


import imp
import importlib


def load(name, path):
    """Load and initialize a module implemented as a Python source file and return its module object"""
    if hasattr(importlib, "machinery"):
        loader = importlib.machinery.SourceFileLoader(name, path)
        return loader.load_module()
    return imp.load_source(name, path)
```

### Accepted 15
```python
from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def working():
    return {"Working"}
```

### Accepted 16
```python
import logging
import numpy
from ..Fragments import Fragments
from ..typing import SpectrumType


logger = logging.getLogger("matchms")


def add_losses(spectrum_in: SpectrumType, loss_mz_from=0.0, loss_mz_to=1000.0) -> SpectrumType:
    """Derive losses based on precursor mass.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    loss_mz_from:
        Minimum allowed m/z value for losses. Default is 0.0.
    loss_mz_to:
        Maximum allowed m/z value for losses. Default is 1000.0.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    precursor_mz = spectrum.get("precursor_mz", None)
    if precursor_mz:
        assert isinstance(precursor_mz, (float, int)), ("Expected 'precursor_mz' to be a scalar number.",
                                                        "Consider applying 'add_precursor_mz' filter first.")
        peaks_mz, peaks_intensities = spectrum.peaks.mz, spectrum.peaks.intensities
        losses_mz = (precursor_mz - peaks_mz)[::-1]
        losses_intensities = peaks_intensities[::-1]
        # Add losses which are within given boundaries
        mask = numpy.where((losses_mz >= loss_mz_from)
                           & (losses_mz <= loss_mz_to))
        spectrum.losses = Fragments(mz=losses_mz[mask],
                                    intensities=losses_intensities[mask])
    else:
        logger.warning("No precursor_mz found. Consider applying 'add_precursor_mz' filter first.")

    return spectrum
```

### Accepted 17
```python
import os
import unittest

import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import torch.nn as nn

from machina.optims import DistributedAdamW


def init_processes(rank, world_size,
                   function, backend='tcp'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank,
                            world_size=world_size)
    function(rank, world_size)


class TestDistributedAdamW(unittest.TestCase):

    def test_step(self):

        def _run(rank, world_size):
            model = nn.Linear(10, 1)
            optimizer = DistributedAdamW(
                model.parameters())

            optimizer.zero_grad()
            loss = model(torch.ones(10).float())
            loss.backward()
            optimizer.step()

        processes = []
        world_size = 4
        for rank in range(world_size):
            p = Process(target=init_processes,
                        args=(rank,
                              world_size,
                              _run))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
```

### Accepted 18
```python
from pytube import YouTube

def download_video(watch_url):
    yt = YouTube(watch_url)
    (yt.streams
        .filter(progressive=True, file_extension='mp4')
        .order_by('resolution')
        .desc()
        .first()
        .download())
```

### Accepted 19
```python
from sys import argv
from PyPDF2 import PdfFileReader, PdfFileWriter
import re


range_pattern = re.compile(r'(\d+)(\.\.|-)(\d+)')
comma_pattern = re.compile('\d+(,\d+)*')
def pages_args_to_array(pages_str):
	groups = range_pattern.search(pages_str)
	if groups:
		start = int(groups.group(1))
		end = int(groups.group(3))
		return list(range(start, end + 1))
	elif comma_pattern.search(pages_str):
		return [int(d) for d in pages_str.split(',')]
	else:
		raise Exception('pages should be like 1,2,3 or 1-3, but was {}'
			.format(pages_str))


if __name__ == '__main__':
	assert(len(argv) > 1), "usage examle:\npython3 selective_merge_pdf.py file1.pdf 1-3 file2.pdf 3,4,10 file1.pdf 50"
	assert(len(argv) % 2 == 1), "invalid arguments; supply page numbers after each pdf name"

	files_names = argv[1::2]
	pages_args = argv[2::2]


	pdf_writer = PdfFileWriter()
	for file_name, pages in zip(files_names, pages_args):
		pdf_reader = PdfFileReader(file_name)
		last_page_index = pdf_reader.getNumPages()
		pages = pages_args_to_array(pages)
		pages_to_add = list(filter(lambda i: i >= 0 and i <= last_page_index, pages))
		for page in pages_to_add:
		    pdf_writer.addPage(pdf_reader.getPage(page - 1))

	with open("merged.pdf", 'wb') as out:
		pdf_writer.write(out)
```

### Accepted 20
```python
"""
Central configuration module of webstr selenium tests.

This module provides configuration options along with default values and
function to redefine values.
"""

# Copyright 2016 Red Hat
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


import logging
import sys


SELENIUM_LOG_LEVEL = logging.INFO
SCHEME = 'https'
PORT = 443
BROWSER = 'Firefox'
BROWSER_VERSION = ''
BROWSER_PLATFORM = 'ANY'
SELENIUM_SERVER = None
SELENIUM_PORT = 4444
BROWSER_WIDTH = 1280
BROWSER_HEIGHT = 1024


def update_value(key_name, value, force=False):
    """
    Update single value of this config module.
    """
    this_module = sys.modules[__name__]
    key_name = key_name.upper()
    # raise AttributeError if we try to define new value (unless force is used)
    if not force:
        getattr(this_module, key_name)
    setattr(this_module, key_name, value)
```

### Accepted 21
```python
def ips_between(start, end):
    calc = lambda n, m: (int(end.split(".")[n]) - int(start.split(".")[n])) * m
    return calc(0, 256 * 256 * 256) + calc(1, 256 * 256) + calc(2, 256) + calc(3, 1)
```

### Accepted 22
```python
from sklearn.linear_model import LogisticRegression
from fightchurn.listings.chap8.listing_8_2_logistic_regression import prepare_data, save_regression_model
from fightchurn.listings.chap8.listing_8_2_logistic_regression import save_regression_summary, save_dataset_predictions

def regression_cparam(data_set_path, C_param):
    X,y = prepare_data(data_set_path)
    retain_reg = LogisticRegression( C=C_param, penalty='l1', solver='liblinear', fit_intercept=True)
    retain_reg.fit(X, y)
    c_ext = '_c{:.3f}'.format(C_param)
    save_regression_summary(data_set_path,retain_reg,ext=c_ext)
    save_regression_model(data_set_path,retain_reg,ext=c_ext)
    save_dataset_predictions(data_set_path,retain_reg,X,ext=c_ext)
```

### Accepted 23
```python
"""
Minimum edit distance computes the cost it takes to get from one string to another string. 
This implementation uses the Levenshtein distance with a cost of 1 for insertions or deletions and a cost of 2 for substitutions.

Resource: https://en.wikipedia.org/wiki/Edit_distance

For example, getting from "intention" to "execution" is a cost of 8.
minimum_edit_distance("intention", "execution")
# 8 
"""
def minimum_edit_distance(source, target):
    n = len(source)
    m = len(target)
    D = {}

    # Initialization
    for i in range(0, n+1):
        D[i,0] = i
    for j in range(0, m+1):
        D[0,j] = j
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            if source[i-1] == target[j-1]:
                D[i,j] = D[i-1, j-1]
            else:
                D[i,j] = min(
                    D[i-1, j] + 1,
                    D[i, j-1] + 1,
                    D[i-1, j-1] + 2
                )

    return D[n-1, m-1]
```

### Accepted 24
```python
import os

def readlinkabs(l):
    """
    Return an absolute path for the destination 
    of a symlink
    """
    if not (os.path.islink(l)):
        return None
    p = os.readlink(l)
    if os.path.isabs(p):
        return p
    return os.path.join(os.path.dirname(l), p)
```

### Accepted 25
```python
import numpy as np
from scipy import ndimage


def erode_value_blobs(array, steps=1, values_to_ignore=tuple(), new_value=0):
    unique_values = list(np.unique(array))
    all_entries_to_keep = np.zeros(shape=array.shape, dtype=np.bool)
    for unique_value in unique_values:
        entries_of_this_value = array == unique_value
        if unique_value in values_to_ignore:
            all_entries_to_keep = np.logical_or(entries_of_this_value, all_entries_to_keep)
        else:
            eroded_unique_indicator = ndimage.binary_erosion(entries_of_this_value, iterations=steps)
            all_entries_to_keep = np.logical_or(eroded_unique_indicator, all_entries_to_keep)
    result = array * all_entries_to_keep
    if new_value != 0:
        eroded_entries = np.logical_not(all_entries_to_keep)
        new_values = new_value * eroded_entries
        result += new_values
    return result
```

### Accepted 26
```python
from helpers import *
     
def test_f_login_andy():
    url = "http://central.orbits.local/rpc.AuthService/Login"
    raw_payload = {"name": "andy","password": "<PASSWORD>"}
    payload = json.dumps(raw_payload)
    headers = {'Content-Type': 'application/json'}

    
    # convert dict to json by json.dumps() for body data. 
    response = requests.request("POST", url, headers=headers, data=payload)
    save_cookies(response.cookies,"cookies.txt")
    
    # Validate response headers and body contents, e.g. status code.
    assert response.status_code == 200

    # print full request and response
    pretty_print_request(response.request)
    pretty_print_response(response)
```

### Accepted 27
```python
# -*- encoding: utf-8 -*-

import multiprocessing as mp
import time
from pudb.remote import set_trace


def worker(worker_id):
    """ Simple worker process"""
    i = 0
    while i < 10:
        if worker_id == 1:  # debug process with id 1
            set_trace(term_size=(80, 24))
        time.sleep(1)  # represents some work
        print('In Process {}, i:{}'.format(worker_id, i))
        i = i + 1


if __name__ == '__main__':
    processes = []
    for p_id in range(2):  # 2 worker processes
        p = mp.Process(target=worker, args=(p_id,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
```

### Accepted 28
```python
def solution(string):
    return string[::-1]
```

### Accepted 29
```python
# -*- coding: utf-8 -*-
"""Utilities common to CIFAR10 and CIFAR100 datasets.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from six.moves import cPickle


def load_batch(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data.

    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.

    # Returns
        A tuple `(data, labels)`.
    """
    with open(fpath, 'rb') as f:
        if sys.version_info < (3,):
            d = cPickle.load(f)
        else:
            d = cPickle.load(f, encoding='bytes')
            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels
```

### Accepted 30
```python
from core.advbase import *
from slot.d import *

def module():
    return Luther

class Luther(Adv):
    a1 = ('cc',0.10,'hit15')

    conf = {}
    conf ['slots.d'] = Leviathan()
    conf['acl'] = """
        `dragon
        `s1
        `s2, seq=5 and cancel
        `s3, seq=5 and cancel or fsc
        `fs, seq=5
    """
    coab = ['Blade', 'Xander', 'Tiki']

if __name__ == '__main__':
    from core.simulate import test_with_argv
    test_with_argv(None, *sys.argv)
```

## 30 Rejected Examples with Reasons

### Rejected 1
**Reason:** SyntaxError (ast.parse failed)
```python
<reponame>MTES-MCT/sparte
from rest_framework_gis import serializers
from rest_framework import serializers as s

from .models import (
    Artificialisee2015to2018,
    Artificielle2018,
    CommunesSybarval,
    CouvertureSol,
    EnveloppeUrbaine2018,
    Ocsge,
    Renaturee2018to2015,
    Sybar...
```

### Rejected 2
**Reason:** No function definition found
```python
from django.contrib import admin
from .models import SearchResult

# Register your models here.
class SearchResultAdmin(admin.ModelAdmin):
    fields = ["query", "heading", "url", "text"]

admin.site.register(SearchResult, SearchResultAdmin)...
```

### Rejected 3
**Reason:** Too long (>2000 chars)
```python
import asyncio
import os
import tempfile
from contextlib import ExitStack
from typing import Text, Optional, List, Union, Dict

from rasa.importers.importer import TrainingDataImporter
from rasa import model
from rasa.model import FingerprintComparisonResult
from rasa.core.domain import Domain
from ...
```

### Rejected 4
**Reason:** SyntaxError (ast.parse failed)
```python
<gh_stars>1-10
class Solution:
    def finalPrices(self, prices: List[int]) -> List[int]:
        res = []
        for i in range(len(prices)):
            for j in range(i+1,len(prices)):
                if prices[j]<=prices[i]:
                    res.append(prices[i]-prices[j])
                  ...
```

### Rejected 5
**Reason:** SyntaxError (ast.parse failed)
```python
<gh_stars>0
# ============================================================================
# FILE: default.py
# AUTHOR: <NAME> <<EMAIL> at g<EMAIL>>
# License: MIT license
# ============================================================================

import re
import typing

from denite.util import...
```

### Rejected 6
**Reason:** SyntaxError (ast.parse failed)
```python
<filename>PyDSTool/core/context_managers.py
# -*- coding: utf-8 -*-

"""Context managers implemented for (mostly) internal use"""

import contextlib
import functools
from io import UnsupportedOperation
import os
import sys


__all__ = ["RedirectStdout", "RedirectStderr"]


@contextlib.contextmanager...
```

### Rejected 7
**Reason:** No function definition found
```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from . import __version__ as app_version

app_name = "pos_kiosk"
app_title = "Pos Kiosk"
app_publisher = "9t9it"
app_description = "Kiosk App"
app_icon = "octicon octicon-file-directory"
app_color = "grey"
app_email = "<EMAIL>"
app_lice...
```

### Rejected 8
**Reason:** SyntaxError (ast.parse failed)
```python
<gh_stars>1-10
from keras import Model, Input
from keras.layers import Dense, concatenate, LSTM, Reshape, Permute, Embedding, Dropout, Convolution1D, Flatten
from keras.optimizers import Adam

from pypagai.models.base import KerasModel


class SimpleLSTM(KerasModel):
    """
    Use a simple lstm ne...
```

### Rejected 9
**Reason:** SyntaxError (ast.parse failed)
```python
<filename>lib/variables/latent_variables/__init__.py
from .fully_connected import FullyConnectedLatentVariable
from .convolutional import ConvolutionalLatentVariable...
```

### Rejected 10
**Reason:** Too long (>2000 chars)
```python
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:
''' PNASNet in PyTorch.
Paper: Progressive Neural Architecture Search
'''

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.backbone_name import BackboneName
from easyai.model.backbone.utility.base_...
```

### Rejected 11
**Reason:** Too long (>2000 chars)
```python
# -*- coding: utf-8 -*-
#  coding=utf-8
import json
import os
import math
import logging
import requests
import time

from map_download.cmd.BaseDownloader import DownloadEngine, BaseDownloaderThread, latlng2tile_terrain, BoundBox


def get_access_token(token):
    resp = None
    request_count = 0
 ...
```

### Rejected 12
**Reason:** SyntaxError (ast.parse failed)
```python
<reponame>vahini01/electoral_rolls
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 23:28:58 2017

@author: dhingratul
"""
import urllib.request
import os
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from bs4 import BeautifulSoup
import ssl
...
```

### Rejected 13
**Reason:** SyntaxError (ast.parse failed)
```python
<gh_stars>0
"""
Experiment summary
------------------
Treat each province/state in a country cases over time
as a vector, do a simple K-Nearest Neighbor between
countries. What country has the most similar trajectory
to a given country?

Plots similar countries
"""

import sys
sys.path.insert(0, '.....
```

### Rejected 14
**Reason:** SyntaxError (ast.parse failed)
```python
<reponame>steven-lang/rational_activations
"""
Rational Activation Functions for MXNET
=======================================

This module allows you to create Rational Neural Networks using Learnable
Rational activation functions with MXNET networks.
"""
import mxnet as mx
from mxnet import initia...
```

### Rejected 15
**Reason:** SyntaxError (ast.parse failed)
```python
<filename>torchflare/criterion/utils.py<gh_stars>1-10
"""Utils for criterion."""
import torch
import torch.nn.functional as F


def normalize(x, axis=-1):
    """Performs L2-Norm."""
    num = x
    denom = torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12
    return num / denom


# Source :...
```

### Rejected 16
**Reason:** No function definition found
```python
"""Tests for the sbahn_munich integration"""


line_dict = {
    "name": "S3",
    "color": "#333333",
    "text_color": "#444444",
}...
```

### Rejected 17
**Reason:** SyntaxError (ast.parse failed)
```python
<reponame>geudrik/hautomation
#! /usr/bin/env python2.7
# -*- coding: latin-1 -*-

from flask import Blueprint
from flask import current_app
from flask import render_template

from flask_login import login_required

homestack = Blueprint("homestack", __name__, url_prefix="/homestack")


@homestack.r...
```

### Rejected 18
**Reason:** Too long (>2000 chars)
```python
"""Forms for RTD donations"""

import logging

from django import forms
from django.conf import settings
from django.utils.translation import ugettext_lazy as _

from readthedocs.payments.forms import StripeModelForm, StripeResourceMixin
from readthedocs.payments.utils import stripe

from .models im...
```

### Rejected 19
**Reason:** Too long (>2000 chars)
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .base import DataReaderBase
from ..tools import COL, _get_dates, to_float, to_int

import pandas as pd
#from pandas.tseries.frequencies import to_offset
from six.moves import cStringIO as StringIO
import logging
import traceback
import datetime

im...
```

### Rejected 20
**Reason:** SyntaxError (ast.parse failed)
```python
<reponame>Vail-qin/Keras-TextClassification
# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/11/2 21:08
# @author  : Mo
# @function:


from keras_textclassification.data_preprocess.text_preprocess import load_json, save_json
from keras_textclassification.conf.path_config import path_mode...
```

### Rejected 21
**Reason:** Too long (>2000 chars)
```python
# Copyright 2014 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

from gpu_tests.gpu_test_expectations import GpuTestExpectations

# See the GpuTestExpectations class for documentation.

class PixelExpec...
```

### Rejected 22
**Reason:** SyntaxError (ast.parse failed)
```python
<filename>examples/p02_budgets/budget_data_ingest/migrations/0001_initial.py
# -*- coding: utf-8 -*-
# Generated by Django 1.11.13 on 2018-06-08 22:54
from __future__ import unicode_literals

from django.conf import settings
import django.contrib.postgres.fields.jsonb
from django.db import migration...
```

### Rejected 23
**Reason:** No function definition found
```python
import setuptools  #enables develop

setuptools.setup(
    name='pysvm',
    version='0.1',
    description='PySVM : A NumPy implementation of SVM based on SMO algorithm',
    author_email="<EMAIL>",
    packages=['pysvm'],
    license='MIT License',
    long_description=open('README.md', encoding='...
```

### Rejected 24
**Reason:** SyntaxError (ast.parse failed)
```python
<gh_stars>1-10
######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: <NAME>
# Date: 1/15/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on an image.
...
```

### Rejected 25
**Reason:** No function definition found
```python
from data_collection.management.commands import BaseXpressDemocracyClubCsvImporter

class Command(BaseXpressDemocracyClubCsvImporter):
    council_id = 'E06000027'
    addresses_name = 'parl.2017-06-08/Version 1/Torbay Democracy_Club__08June2017.tsv'
    stations_name = 'parl.2017-06-08/Version 1/To...
```

### Rejected 26
**Reason:** Contains 0 functions (want exactly 1 clean function)
```python
from sys import maxsize


class Contact:

    def __init__(self, fname=None, mname=None, lname=None, nick=None, title=None, comp=None, addr=None,
                 home=None, mobile=None, work=None, fax=None, email1=None, email2=None, email3=None,
                 homepage=None, bday=None, bmonth=Non...
```

### Rejected 27
**Reason:** Too long (>2000 chars)
```python
##########################################################################
#
#  Copyright (c) 2010-2012, Image Engine Design Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  ...
```

### Rejected 28
**Reason:** SyntaxError (ast.parse failed)
```python
<filename>rlpy/Domains/Pacman.py
"""Pacman game domain."""
from rlpy.Tools import __rlpy_location__
from .Domain import Domain
from .PacmanPackage import layout, pacman, game, ghostAgents
from .PacmanPackage import graphicsDisplay
import numpy as np
from copy import deepcopy
import os
import time

_...
```

### Rejected 29
**Reason:** Contains 0 functions (want exactly 1 clean function)
```python
from zeit.cms.i18n import MessageFactory as _
import zope.interface
import zope.schema


class IGlobalSettings(zope.interface.Interface):
    """Global CMS settings."""

    default_year = zope.schema.Int(
        title=_("Default year"),
        min=1900,
        max=2100)

    default_volume = zop...
```

### Rejected 30
**Reason:** SyntaxError (ast.parse failed)
```python
<filename>abc/abc165/abc165e.py
N, M = map(int, input().split())

for i in range(1, M + 1):
    if i % 2 == 1:
        j = (i - 1) // 2
        print(1 + j, M + 1 - j)
    else:
        j = (i - 2) // 2
        print(M + 2 + j, 2 * M + 1 - j)...
```
