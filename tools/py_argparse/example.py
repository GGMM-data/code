import argparse

parser = argparse.ArgumentParser()

parser.add_argument('test', type=int)
parser.add_argument('--t', type=int)

args = parser.parse_args(['3', '--t', '4'])

print(args)
