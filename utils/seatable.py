import datetime
import os
from pprint import pprint as pp
from typing import List

from seatable_api import Base

ssl_aug_api_token = '669cdc2fc382cefb698d4c629dc2164e1f6772c5'
server_url = 'https://cloud.seatable.cn'
base = Base(ssl_aug_api_token, server_url)
base.auth()

tag_choices = [
    'mlp', 'sbn', 'cos',
    'wp', 'nowd',
    'sym',
]


def record_dt(dd):
    dd['last_upd'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')


def fill_explore_table(abs_path=None, rid=None, **kwargs):
    t_name = 'explore'
    tags = []
    for k, v in kwargs.items():
        if k in tag_choices and v:
            tags.append(k)
            kwargs.pop(k)
    
    dd = dict(tags=tags, **kwargs)
    if abs_path is not None:
        dd['abs_path'] = abs_path
        dd['exp'] = '/'.join(abs_path.split('/')[-2:])
    record_dt(dd)
    
    if rid is None:
        return base.append_row(t_name, dd)['_id']
    else:
        return base.update_row(t_name, rid, dd)['success']
