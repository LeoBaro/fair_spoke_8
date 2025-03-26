import webdataset as wds
import argparse

def filter_tar(input_tar, output_tar, N):
    dataset = wds.WebDataset(input_tar).decode()
    
    with wds.TarWriter(output_tar) as sink:
        for i, sample in enumerate(dataset):
            if i >= N:
                break
            sink.write(sample)
    
    print(f"Filtered tar file saved as {output_tar}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract first N samples from a tar file using webdataset.")
    parser.add_argument("input_tar", type=str, help="Name of the input tar file")
    parser.add_argument("output_tar", type=str, help="Name of the output tar file")
    parser.add_argument("N", type=int, help="Number of samples to keep")
    
    args = parser.parse_args()
    filter_tar(args.input_tar, args.output_tar, args.N)
