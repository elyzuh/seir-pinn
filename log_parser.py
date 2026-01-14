#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import glob
import numpy as np
import pandas as pd

## Find the best performance in sets of logs
# extract the values from log
def extract_tst_from_log(filename):
    # empty file
    lines = open(filename).readlines()
    if len(lines) < 1:
        return 1e10, 1e10, -1, filename
    line = lines[-1]
    # invalid or NaN
    if not line.startswith('test rse'):
        return 1e10, 1e10, -1, filename
    fields = line.split('|')
    tst_rse = float(fields[0].split()[2])
    tst_rae = float(fields[1].split()[2])
    tst_cor = float(fields[2].split()[2])
    return tst_rse, tst_rae, tst_cor, filename

def format_logs(raw_expression):
    val_filenames = []
    Results = {}
    for num in [1, 2, 4]:
        if num not in Results.keys():
            Results[num] = {}

        expressions = raw_expression.format(num)
        filenames = glob.glob(expressions)
        if len(filenames) == 0:
            Results[num]["rmse"] = None
            Results[num]["rae"] = None
            Results[num]["corr"] = None
            Results[num]["best_log"] = "No logs found"
            continue

        tuple_list = [extract_tst_from_log(filename) for filename in filenames]
        rse_list, rae_list, cor_list, log_files = zip(*tuple_list)

        index = np.argmin(rse_list)  # best = lowest RSE

        best_rse = rse_list[index]
        best_rae = rae_list[index]
        best_cor = cor_list[index]
        best_log_file = log_files[index]

        print('horizon:{:2d} | rmse: {:.4f} | rae: {:.4f} | corr: {:.4f} | best_log: {}'.format(
            num, best_rse, best_rae, best_cor, best_log_file))

        Results[num]["rmse"] = '{:.4f}'.format(best_rse)
        Results[num]["rae"] = '{:.4f}'.format(best_rae)
        Results[num]["corr"] = '{:.4f}'.format(best_cor)
        Results[num]["best_log"] = best_log_file  # optional: keep in results if needed

    ResultsDf = pd.DataFrame.from_dict(Results)
    ResultsDf = ResultsDf[[1, 2, 4]]
    ResultsDf = ResultsDf.reindex(['rmse', 'rae', 'corr'])
    print("\nSummary Table:")
    print(ResultsDf)


if __name__ == '__main__':
    for data in ['hhs']:        
        print("--------------------Data:", data)
        # cnnrnn_res_epi
        print('*' * 40)
        print("CNNRNN_Res_Epi", "-", data)
        format_logs('./log/cnnrnn_res_epi/cnnrnn_res_epi.%s.hid-*.drop-*.w-*.h-{}.ratio-*.res-*.lam-*.out' % (data))

        print("")
        print("")