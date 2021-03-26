import os
import re

def yield_paramed_matches(file_format, raw_params, res_dir):
    params = raw_params.copy()
    for name, value in params.items():
        if value is None:
            params[name] = f'(?P<{name}>[0-9.]+)'
    params_str = '_'.join([f'{key}={value}' for key, value in params.items()])
    file_pattern = re.compile(file_format.format(params_str).replace('.', r'\.'))
    for name in os.listdir(res_dir):
        match = re.search(file_pattern, name)
        if match is not None:
            yield match
