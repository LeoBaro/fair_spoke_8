import argparse
from made.data_pipeline.utils import collect_tar_files
from made.data_pipeline.data.datacomp_handler import decode_webdataset, get_next_batch
import matplotlib.pyplot as plt

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--web-dataset-folder", type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = cli()
    tar_files = collect_tar_files(args.web_dataset_folder, recursive=True)

    dataset = decode_webdataset(
        tar_files,
        get_images=False,
        get_captions=False,
        batch_size=10000
    )
    dataset = iter(dataset)

    count = 0
    checked_uids = {}
    for batch in dataset:
        print(f'\rCounter: {count}', end='', flush=True)
        count += len(batch[0])
        uids = batch[0]
        for uid in uids:
            if uid not in checked_uids.keys():
                checked_uids[uid] = 1
            else:
                checked_uids[uid] += 1

    duplicates = {uid: count for uid, count in checked_uids.items() if count > 1}
    print(f"Number of duplicates: {len(duplicates)}")
    
    if len(duplicates) == 0:
        print("No duplicates found")
        exit(0)
    
    import matplotlib.pyplot as plt
    first_duplicate_uid = list(duplicates.keys())[0]
    fig, ax = plt.subplots(1, duplicates[first_duplicate_uid], figsize=(12, 12))
    dataset = decode_webdataset(
        tar_files,
        get_images=True,
        get_captions=True,
        batch_size=10000
    )
    dataset = iter(dataset)
    iax = 0
    for batch in dataset:
        uids = batch[0]
        for uid in uids:
            if uid == first_duplicate_uid:
                print(f"Duplicate found: {uid}")
                ax[iax].imshow(batch[1][0])
                ax[iax].set_title(f"{uid} \n{batch[2][0]}")
                iax += 1
    plt.show()
