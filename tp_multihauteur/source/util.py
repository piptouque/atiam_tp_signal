
import numpy as np
import collections.abc
import matplotlib.pyplot as plt


def dict_merge(*args, add_keys=True):
    # from: https://gist.github.com/angstwad/bf22d1822c38a92ec0a9#gistcomment-3305932
    assert len(args) >= 2, "dict_merge requires at least two dicts to merge"
    rtn_dct = args[0].copy()
    merge_dicts = args[1:]
    for merge_dct in merge_dicts:
        if add_keys is False:
            merge_dct = {key: merge_dct[key] for key in set(
                rtn_dct).intersection(set(merge_dct))}
        for k, v in merge_dct.items():
            if not rtn_dct.get(k):
                rtn_dct[k] = v
            elif k in rtn_dct and type(v) != type(rtn_dct[k]):
                raise TypeError(
                    f"Overlapping keys exist with different types: original is {type(rtn_dct[k])}, new value is {type(v)}")
            elif isinstance(rtn_dct[k], dict) and isinstance(merge_dct[k], collections.abc.Mapping):
                rtn_dct[k] = dict_merge(
                    rtn_dct[k], merge_dct[k], add_keys=add_keys)
            elif isinstance(v, list):
                for list_value in v:
                    if list_value not in rtn_dct[k]:
                        rtn_dct[k].append(list_value)
            else:
                rtn_dct[k] = v
    return rtn_dct


def next_pow2(n: np.int32) -> np.int32:
    # from: https://stackoverflow.com/a/1322548
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    return n


def F_plot1(x_v, y_v, labelX, labelY):
    plt.plot(x_v, y_v)
    plt.xlabel(labelX)
    plt.ylabel(labelY)
    plt.grid(True)
    return


def F_plot2(data_m, col_v=np.zeros(0), row_v=np.zeros(0), labelCol='', labelRow=''):
    plt.imshow(data_m, origin='lower', aspect='auto', extent=[
               row_v[0], row_v[-1], col_v[0], col_v[-1]], interpolation='nearest')
    plt.colorbar()
    plt.set_cmap('gray_r')
    plt.xlabel(labelRow)
    plt.ylabel(labelCol)
    plt.grid(True)
    return
