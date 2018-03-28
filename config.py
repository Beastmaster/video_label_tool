'''
Read configuration file
'''


import json



def decode_config(fname):
    structures = {}
    try:
        with open(fname,'r') as fp:
            infos = fp.read()
            structures = json.loads(infos)
        return structures
    except:
        return {}




if __name__ == "__main__":
    res = decode_config('config.json')
    res['abc']
    print(res)

