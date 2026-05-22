import os
import argparse
import wfdb
import numpy as np
import torch
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Process WFDB records to tensors")
    parser.add_argument("--input_dir", "-i", type=str, default="training", help="Directory containing WFDB files (.hea/.dat)")
    parser.add_argument("--output", "-o", type=str, default="dataset/samples_alarms.pt", help="Output .pt file path")
    return parser.parse_args()


# signal name
type_s = ['I', 'II', 'III', 'V', 'aVF', 'aVL', 'aVR', 'MCL', 'RESP', 'PLETH', 'ABP']
type_s = dict([(y, x) for x, y in enumerate(type_s)])
type_result = ['False alarm', 'True alarm']
# alarm type
type_ill = ['Asystole', 'Bradycardia', 'Tachycardia', 'Ventricular_Tachycardia', 'Ventricular_Flutter_Fib']
type_ill = dict([(y, x) for x, y in enumerate(type_ill)])
ys = []
alarms = []
samples = []

def main():
    args = parse_args()
    input_dir = args.input_dir
    output_path = args.output

    files = os.listdir(input_dir)

    records = []
    # names of all samples
    for f in files:
        if f.endswith('.hea'):
            records.append(f[:-4])

    count = 0
    for f in records:
        y = torch.zeros(1)
        alarm = torch.zeros(5)
        sample = torch.zeros(11, 75000)
        path = os.path.join(input_dir, f)

    # load signal
    record = wfdb.rdrecord(path)
    sig_len = len(record.__dict__['sig_name'])
    if record.__dict__['comments'][1] == 'True alarm':
        y += 1
    alarm[type_ill[record.__dict__['comments'][0]]] += 1
    for i in range(sig_len):

        seqs = record.__dict__['p_signal'][:, i]
        # set the signal to zero value if signal is nan
        if np.isnan(seqs).all():
            seqs[np.isnan(seqs)] = 0

        # use forward fill and backward fill for missing values imputation
        elif np.isnan(seqs).any():
            seqs = pd.Series(seqs).fillna(method='ffill').fillna(method='bfill').values

        sample[type_s[record.__dict__['sig_name'][i]], :] = torch.from_numpy(seqs[:75000])

    ys.append(y)
    alarms.append(alarm)
    samples.append(sample)


    ys = torch.stack(ys)
    alarms = torch.stack(alarms)
    samples = torch.stack(samples)


    print(ys.sum(0))
    print(alarms.sum(0))
    print(samples.shape)

    # transfer data file to binary
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    torch.save((samples, alarms, ys), output_path)


if __name__ == "__main__":
    main()