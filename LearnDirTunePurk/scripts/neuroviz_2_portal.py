import argparse
import os
import pickle



def neurons_2_unit_info(neurons):
    unit_info = {'channel': [],
                 'spiketimes': [],
                 'snr': [],
                 'template': []
                }
    for n in neurons:
        # If this is a PC then add its CS as a separate unit
        if n['type__'] == "NeurophysToolbox.PurkinjeCell":
            try:
                unit_info['channel'].append(n['channel_id__'][0])
            except IndexError:
                unit_info['channel'].append(n['channel_id__'])
            unit_info['spiketimes'].append(n['cs_spike_indices__'] / n['sampling_rate__'])
            if 'cs_snr' in n:
                unit_info['snr'].append(n['cs_snr'])
            if 'cs_template' in n:
                unit_info['template'].append(n['cs_template'])
        try:
            unit_info['channel'].append(n['channel_id__'][0])
        except IndexError:
            unit_info['channel'].append(n['channel_id__'])
        unit_info['spiketimes'].append(n['spike_indices__'] / n['sampling_rate__'])
        if 'snr' in n:
            unit_info['snr'].append(n['snr'])
        if 'template' in n:
            unit_info['template'].append(n['template'])
    return unit_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("viz_dir")
    parser.add_argument("save_dir")
    args = parser.parse_args()
    for fname in os.listdir(args.viz_dir):
        full_name = os.path.join(args.viz_dir, fname)
        if os.path.isdir(full_name):
            continue
        with open(full_name, 'rb') as fp:
            neurons = pickle.load(fp)
        unit_info = neurons_2_unit_info(neurons)
        portal_name = fname.replace("viz", "portal")
        with open(os.path.join(args.portal_dir, portal_name), 'wb') as fp:
            pickle.dump(unit_info, fp)