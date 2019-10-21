#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals
import os

AUTHOR = 'Benoit Olive'
SITENAME = 'Benoit\'s Blog'
HOME_COVER = 'https://cdn.pixabay.com/photo/2017/04/23/19/30/earth-2254769_960_720.jpg'
SITEURL = 'https://Bolive84.github.io/Blog'
PATH = 'content'
STATIC_PATHS = ['images']
TIMEZONE = 'America/Toronto'
DEFAULT_LANG = 'en'
DEFAULT_PAGINATION = 10
THEME = 'theme'
cover = 'https://cdn.pixabay.com/photo/2018/07/03/07/09/block-chain-3513216_960_720.jpg'
# set to False for Production, True for Development
if os.environ.get('PELICAN_ENV') == 'DEV':
    RELATIVE_URLS = True
else:
    RELATIVE_URLS = False