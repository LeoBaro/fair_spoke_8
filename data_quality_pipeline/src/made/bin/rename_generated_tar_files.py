import argparse
from made.data_pipeline.utils import rename_tar_files

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    return parser.parse_args()

def main(args):
    renamed_files = rename_tar_files(args.input_dir)
    print(renamed_files)
    
if __name__ == "__main__":
    main(cli())