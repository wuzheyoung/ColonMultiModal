# -*- coding: utf-8 -*-
"""Inspect exp_v2_validation_suite_T.json structure (quick audit helper)."""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

d = json.load(open('logs/exp_v2_validation_suite_T.json', encoding='utf-8'))

def walk(o, prefix='', depth=0):
    if depth > 5:
        return
    if isinstance(o, dict):
        for k in list(o)[:40]:
            v = o[k]
            if isinstance(v, (dict, list)):
                print(prefix + k, type(v).__name__, len(v))
                walk(v, prefix + '  ', depth + 1)
            else:
                print(prefix + k, '=', str(v)[:100])
    elif isinstance(o, list) and o:
        print(prefix + '[0] =', str(o[0])[:100])
        if isinstance(o[0], (dict, list)):
            walk(o[0], prefix + '  ', depth + 1)

walk(d)
