import os
import paramiko
import pandas as pd
import gzip

'''
This module is aimed at reading JSON data files, either locally or from a remote
host. The data files are not exactly JSON, they're files in which each line is a
JSON object, thus making up a row of data, and each key of the JSON strings
referring to a column.
'''

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
        print('{:.4g}MB read, {} tweets unpacked.'.format(chunk_size*10**-6,
                                                          nr_tweets))
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
            final_line = f.readline()
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
