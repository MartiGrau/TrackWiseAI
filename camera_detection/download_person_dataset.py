import os
import argparse
import opendatasets as od

def download_dataset(kaggle_url, output_dir):
    """Download the dataset from Kaggle and extract it."""
    # Using opendatasets let's download the data sets
    od.download(dataset_id_or_url=kaggle_url, data_dir=output_dir, force=True)

def main():
    parser = argparse.ArgumentParser(description="Download person dataset.")
    parser.add_argument('--kaggle_url', type=str, default='https://www.kaggle.com/fareselmenshawii/human-dataset', help='Kaggle dataset URL')
    parser.add_argument('--output_dir', type=str, default='/hdd1/Datasets/Public/Raw/Human_Dataset', help='Directory to save the dataset')

    args = parser.parse_args()
    download_dataset(args.kaggle_url, args.output_dir)

if __name__ == "__main__":
    main()