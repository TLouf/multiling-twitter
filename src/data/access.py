'''
This module is aimed at reading JSON data files, either locally or from a remote
host. The data files are not exactly JSON, they're files in which each line is a
JSON object, thus making up a row of data, and in which each key of the JSON
strings refers to a column.
'''
import os
import gzip
import logging
import paramiko
import pandas as pd
import numpy as np

LOGGER = logging.getLogger(__name__)


def yield_tweets_access(tweets_files_paths, tweets_res=None):
    '''
    Yields what we call an access to a tweets' DataFrame, which can either be
    the DataFrame directly if a list `tweets_res` of them is supplied, or the
    arguments of `read_json_wrapper`. The functions applied in a loop started
    from this generator then must have as an argument a "get_df" function to
    finally get a DataFrame (see more detail in comments below).
    Unfortunately we can't make this "get_df" function part of the yield here,
    as the function needs to be pickable (so declared globally) for later use in
    a multiprocessing context.
    '''
    if tweets_res is None:
        # Here get_df = lambda x: read_json_wrapper(*x).
        for file_path in tweets_files_paths:
            for chunk_start, chunk_size in chunkify(file_path, size=1e9):
                yield (file_path, chunk_start, chunk_size)
    else:
        # In this case get_df = lambda x: x is to be used
        for tweets_df in tweets_res:
            yield tweets_df


def filter_df(raw_tweets_df, cols=None, dfs_to_join=None):
    '''
    Filters `raw_tweets_df` via inner joins with a list of dataframes
    `dfs_to_join`, each of which must have their index corresponding to a column
    of `raw_tweets_df`. Can also choose to keep only some columns, with the list
    `cols`.
    '''
    if dfs_to_join is None:
        dfs_to_join = []
    filtered_tweets_df = raw_tweets_df.copy()
    if cols is None:
        cols = filtered_tweets_df.columns.values
    for df in dfs_to_join:
        filtered_tweets_df = filtered_tweets_df.join(df, on=df.index.name,
                                                     how='inner')
    new_nr_tweets = filtered_tweets_df.shape[0]
    LOGGER.info(f'{new_nr_tweets} tweets remaining after filters.')
    filtered_tweets_df = filtered_tweets_df.loc[:, cols]
    return filtered_tweets_df


def read_data(tweets_file_path, chunk_start, chunk_size, dfs_to_join=None,
              cols=None, ssh_domain=None, ssh_username=None):
    '''
    Reads the JSON file at `tweets_file_path` starting at the byte `chunk_start`
    and reading `chunk_size` bytes, and dumps it into a DataFrame. Then the
    tweets DataFrame is filtered via inner joins with a list of dataframes
    `dfs_to_join`.
    '''
    if dfs_to_join is None:
        dfs_to_join = []
    raw_tweets_df = read_json_wrapper(
        tweets_file_path, chunk_start, chunk_size, ssh_domain=ssh_domain,
        ssh_username=ssh_username)
    return filter_df(raw_tweets_df, cols=cols, dfs_to_join=dfs_to_join)


def yield_sftp(file_path, ssh_domain, ssh_username):
    '''
    Yields a SFTP file handler of the file located in 'file_path' on the server
    with domain 'ssh_domain', to which you connect with your user name
    'ssh_username'. It is assumed you have done ssh-copy-id on the remote host
    (so `load_system_host_keys` actually loads something).
    '''
    with paramiko.client.SSHClient() as ssh_client:
        ssh_client.load_system_host_keys()
        ssh_client.connect(ssh_domain, username=ssh_username)
        sftp_client = ssh_client.open_sftp()
        with sftp_client.file(file_path, mode='r') as f:
            yield f


# Better to separate generators (functions with yield) and regular functions
# (terminating with return).
def return_json(file_path, ssh_domain=None, ssh_username=None,
                compression='infer'):
    '''
    Returns a DataFrame from a local or remote json file. Not recommended for
    large data files.
    '''
    is_local = os.path.exists(file_path)
    if is_local:
        data = pd.read_json(file_path, lines=True, compression=compression)

    else:
        # Equivalent to with, the generator contains only one object, the sftp
        # file object, and it actually closes the file when the loop is over.
        for f in yield_sftp(file_path, ssh_domain, ssh_username):
            # Here data is a DataFrame, so the return can be made outside
            # of the file context.
            data = pd.read_json(f, lines=True, compression=compression)
    return data


def yield_json(file_path, ssh_domain=None, ssh_username=None, chunk_size=1000,
               compression='infer'):
    '''
    Yields a JsonReader from a local or remote json file, reading it it chunks.
    This is more suitable to larger files than `return_json`, however it can't
    be parallelized because it would involve a generator of file handles, which
    can't be serialized so this can't be used with `multiprocessing`.
    '''
    is_local = os.path.exists(file_path)
    if is_local:
        data = pd.read_json(file_path, lines=True, chunksize=chunk_size,
                            compression=compression)
        for raw_df in data:
            yield raw_df

    else:
        # Equivalent to with, the generator contains only one object, the sftp
        # file object, and it actually closes the file when the loop is over.
        for f in yield_sftp(file_path, ssh_domain, ssh_username):
            data = pd.read_json(f, lines=True, chunksize=chunk_size,
                                compression=compression)
            # Here data is a JsonReader, so the yield can't be made outside
            # of the file context, otherwise the file is closed and the
            # data can't be accessed.
            for raw_df in data:
                yield raw_df



def yield_gzip(file_path, ssh_domain=None, ssh_username=None):
    '''
    Yields a gzip file handler from a remote or local directory.
    '''
    is_local = os.path.exists(file_path)
    if is_local:
        with gzip.open(file_path, 'rb') as unzipped_f:
            yield unzipped_f
    else:
        for f in yield_sftp(file_path, ssh_domain, ssh_username):
            with gzip.open(f, 'rb') as unzipped_f:
                yield unzipped_f


def read_json_wrapper(file_path, chunk_start, chunk_size, ssh_domain=None,
                      ssh_username=None):
    '''
    Reads a DataFrame from the json file in 'file_path', starting at the byte
    'chunk_start' and reading 'chunk_size' bytes.
    '''
    for f in yield_gzip(file_path, ssh_domain=ssh_domain,
                        ssh_username=ssh_username):
        f.seek(chunk_start)
        lines = f.read(chunk_size)
        raw_tweets_df = pd.read_json(lines, lines=True)
        nr_tweets = len(raw_tweets_df)
        LOGGER.info(f'{chunk_size*10**-6:.4g}MB read, {nr_tweets} tweets '
                    'unpacked.')
        return raw_tweets_df


def chunkify(file_path, size=5e8, ssh_domain=None, ssh_username=None):
    '''
    Generator going through a json file located in 'file_path', and yielding the
    chunk start and size of (approximate byte) size 'size'. Since we want to
    read lines of data, the function ensures that the end of the chunk
    'chunk_end' is at the end of a line.
    '''
    for f in yield_gzip(file_path, ssh_domain=ssh_domain,
                        ssh_username=ssh_username):
        chunk_end = f.tell()
        while True:
            chunk_start = chunk_end
            # Seek 'size' bytes ahead, relatively to where we are now (second
            # argument = 1)
            f.seek(int(size), 1)
            # Read a line at this point, that is, read until a '\n' is
            # encountered:
            f.readline()
            # Put the end of the chunk at the end of this line:
            chunk_end = f.tell()
            # If the end of the file is reached, f.tell() returns
            # the last byte, even if we keep seeking forward.
            yield chunk_start, chunk_end-chunk_start
            # Because of readline, we'll always read some bytes more than
            # 'size', if it's not the case it means we've reached the end of the
            # file.
            if chunk_end - chunk_start < size:
                break

# Might be better on larger files, but it's not on CH (compressed 2.4GB)
def test_chunkify(file_path, size=5e8, uncompressed_size=None, ssh_domain=None,
                  ssh_username=None):
    '''
    Generator going through a json file located in 'file_path', and yielding the
    chunk start and size of (approximate byte) size 'size'. Since we want to
    read lines of data, the function ensures that the end of the chunk
    'chunk_end' is at the end of a line.
    '''
    for f in yield_gzip(file_path, ssh_domain=ssh_domain,
                        ssh_username=ssh_username):
        if not uncompressed_size:
            uncompressed_size = f.seek(0, 2)
        for chunk_start in np.arange(0, uncompressed_size, size):
            yield chunk_start, size


def test_read_json_wrapper(file_path, chunk_start, chunk_size, ssh_domain=None,
                      ssh_username=None):
    '''
    Reads a DataFrame from the json file in 'file_path', starting at the byte
    'chunk_start' and reading 'chunk_size' bytes.
    '''
    for f in yield_gzip(file_path, ssh_domain=ssh_domain,
                        ssh_username=ssh_username):
        f.seek(int(chunk_start))
        # readlines reads at least chunk_size, so if we seek chunk_size forward,
        # to get to the next line we need to readline where we are to finish the
        # current line, and then we can readlines(chunk_size), and we'll be
        # starting at the next tweet.
        if chunk_start > 0:
            f.readline()
        # Problem: the generator passing us chunk_start doesn't see that we
        # actually started at a higher byte than chunk_start, so if there are
        # many chunks we might get duplicate lines.
        lines_list = f.readlines(int(chunk_size))
        lines = b''.join(lines_list)
        raw_tweets_df = pd.read_json(lines, lines=True)
        nr_tweets = len(raw_tweets_df)
        print(f'{chunk_size*10**-6:.4g}MB read, {nr_tweets} tweets unpacked.')
        return raw_tweets_df
