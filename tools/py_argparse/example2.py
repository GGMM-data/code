import argparse

def parse_args():
    parser = argparse.ArgumentParser("input parameters")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    args_list = parser.parse_args()
    return args_list

	
def main(args_list):
	print(args_list.batch_size)


if __name__ == "__main__":
    args_list = parse_args()
    main(args_list)

