import argparse
import os
import json
from pathlib import Path


def check_if_postive_int(value):
    _value = int(value)
    if _value <= 0:
        raise argparse.ArgumentTypeError('%s This is an not an int value' % value)
    return _value


parser = argparse.ArgumentParser()
parser.add_argument('-lst', '--actions', nargs='+', help="list of actions, Example --> pick node, wave", required=True)
# parser.add_argument("-sbf", "--subfolder_numbers", help="number of subfolders, must be int", required=True,
#                     type=check_if_postive_int)
parser.add_argument("-dir", "--pat", help="Parent directory for data, must be in string", required=True, type=str)
parser.add_argument("-len", "--video_length", help='The length of the video in frames', required=True,
                    type=check_if_postive_int)
args = parser.parse_args()


def create_dir(actions, data_dir):
    i = 1
    if not os.path.exists(data_dir):
        data_dir = os.path.join(data_dir)

    while os.path.exists(data_dir):
        data_dir = ('{}_%s'.format(args.pat) % i)
        i += 1
        data_dir = os.path.join(data_dir)

    for action in actions:
        # for folder in range(subfolder_numbers):Commented this out as there is no need for subfolder
        try:
            os.makedirs(os.path.join(data_dir, action))  # str(folder)
        except:
            pass
        finally:
            # w = '{}.iams'.format(data_dir)
            iamsdict = {'actions': args.actions, 'video_length': args.video_length,
                        'Data_Directory': data_dir,
                        'Data_Subfolder': str(Path("data/{}".format(data_dir)))}
            # iamsdict['subfolder_length'] = args.subfolder_numbers
            w = 'iamsfixation.iams'
            if not os.path.exists(w):
                w = os.path.join(w)
                f = open(w, 'w')
                json.dump(iamsdict, f)
                f.close()
            os.path.exists(w)
            f = open(w, 'r+')
            f.seek(0)
            json.dump(iamsdict, f)
            f.truncate()
            f.close()
    print(data_dir)


if __name__ == '__main__':
    create_dir(args.actions, args.pat)
