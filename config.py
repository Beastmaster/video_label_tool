'''
Read configuration file
'''


import json



def decode_config(fname):
    structures = {}
    with open(fname,'r') as fp:
        infos = fp.read()
        structures = json.loads(infos)
    return structures




if __name__ == "__main__":
    res = decode_config('config.json')
    res['abc']
    print(res)

