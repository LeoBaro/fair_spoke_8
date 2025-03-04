import argparse
def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards_path", required=True, nargs="+")
    parser.add_argument("--ray-address", type=str, required=True)
    parser.add_argument("--log-folder", type=str, required=True)
    parser.add_argument("--output-folder", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=False, default=None)
    return parser.parse_args()