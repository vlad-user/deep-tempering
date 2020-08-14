import os
import tarfile
import random
import sys
from six.moves import urllib
import urllib.request
import pickle
import gzip
import zipfile

import numpy as np


EMNIST_URL = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip'


def get_emnist_letters(fname='emnist-letters-from-src.pkl'):
    _maybe_download_emnist()
    dirname = os.path.dirname(os.path.abspath(__file__))
    dirname = os.path.join(dirname, 'data')
    fname = os.path.join(dirname, fname)
    if os.path.exists(fname):
        with open(fname, 'rb') as fo:
            obj = pickle.load(fo)
        x_train = obj['x_train']
        y_train = obj['y_train']
        x_test = obj['x_test']
        y_test = obj['y_test']
        x_valid = obj['x_valid']
        y_valid = obj['y_valid']


    else:
        gzip_path = os.path.dirname(os.path.abspath(__file__))
        gzip_path = os.path.join(gzip_path, 'data')
        dst_path = os.path.join(gzip_path, 'emnist_extracted')
        gzip_path = os.path.join(gzip_path, 'gzip.zip')

        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        dst_path = os.path.join(dst_path)
        fnames_dict = {
            'x_test': 'emnist-letters-test-images-idx3-ubyte.gz',
            'y_test': 'emnist-letters-test-labels-idx1-ubyte.gz',
            'x_train': 'emnist-letters-train-images-idx3-ubyte.gz',
            'y_train': 'emnist-letters-train-labels-idx1-ubyte.gz'
        }
        fullpaths = {k: os.path.join(dst_path, 'gzip', v) for k, v in fnames_dict.items()}

        for attempt in range(5):
            try:
                zip_ref = zipfile.ZipFile(gzip_path)
                zip_ref.extractall(dst_path)
                zip_ref.close()
                break
            except zipfile.BadZipFile:

                if attempt == 4:
                    err_msg = ("Can't download EMNIST dataset. Try "
                               "downloading EMNIST dataset manually and place the "
                               "gzip.zip file to the parallel-tempring/simulator/data "
                               "folder.")
                    raise ValueError(err_msg)
                os.remove(gzip_path)
                _maybe_download_emnist()

        def _read4bytes(bytestream):
            dtype = np.dtype(np.uint32).newbyteorder('>')
            return np.frombuffer(bytestream.read(4), dtype=dtype)[0]

        def ungzip_data(fname):
            with gzip.GzipFile(fname, 'r') as fo:
                magic = _read4bytes(fo)
                n_images = _read4bytes(fo)
                n_rows = _read4bytes(fo)
                n_cols = _read4bytes(fo)
                buf = fo.read()
                data = np.frombuffer(buf, dtype=np.uint8)
            return data.reshape(n_images, n_rows, n_cols, 1)

        def ungzip_labels(fname):
            with gzip.GzipFile(fname, 'r') as fo:
                magic = _read4bytes(fo)
                n_labels = _read4bytes(fo)
                buf = fo.read()
                data = np.frombuffer(buf, dtype=np.uint8)
            return data

        x_train = ungzip_data(fullpaths['x_train'])
        y_train = ungzip_labels(fullpaths['y_train'])
        x_test = ungzip_data(fullpaths['x_test'])
        y_test = ungzip_labels(fullpaths['y_test'])

    return x_train, y_train, x_test, y_test


def _maybe_download_emnist():
    filepath = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(filepath, 'data')
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    filename = EMNIST_URL.split('/')[-1]
    filepath = os.path.join(filepath, filename)

    if os.path.exists(filepath):
        return

    def _progress(count, block_size, total_size):
        buff = '\r>> Downloading EMNIST %s %.1f%%' % (filename,
                                                      float(count * block_size) / float(total_size) * 100.0)
        sys.stdout.write(buff)
        sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(
        EMNIST_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded EMNIST dataset', statinfo.st_size, 'bytes.')