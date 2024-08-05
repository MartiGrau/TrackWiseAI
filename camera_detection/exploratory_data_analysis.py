import fiftyone as fo
import fiftyone.zoo as foz
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Function to load dataset and filter it
def load_and_filter_dataset(classes, max_samples):
    datasets = foz.load_zoo_dataset('coco-2017', splits=('train', 'validation', 'test'), classes=classes, progress=True, max_samples=max_samples)

    for sample in datasets:
        if sample.ground_truth is None:
            continue

        # Filter detections to only include specified classes
        detections = [detection for detection in sample.ground_truth.detections if detection.label in classes]
        sample.ground_truth.detections = detections
        sample.save()

    return datasets

# Function to export dataset splits
def export_splits(datasets, export_dir, classes):
    for split in ['train', 'validation', 'test']:
        split_view = datasets.match_tags(split)
        split_view.export(
            export_dir=export_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            label_field='ground_truth',
            split=split,
            classes=classes,
        )

# Main function to parse arguments and run the script
def main():
    parser = argparse.ArgumentParser(description="Load and export filtered COCO-2017 dataset for YOLOv5.")
    parser.add_argument('--classes', type=str, nargs='+', default=['person'], help='Classes to filter in the dataset')
    parser.add_argument('--max_samples', type=int, default=10000, help='Maximum number of samples to load')
    parser.add_argument('--export_dir', type=str, default='./yolov5-coco-datasets', help='Directory to export the dataset')

    args = parser.parse_args()

    datasets = load_and_filter_dataset(classes=args.classes, max_samples=args.max_samples)
    export_splits(datasets, export_dir=args.export_dir, classes=args.classes)

if __name__ == "__main__":
    main()
