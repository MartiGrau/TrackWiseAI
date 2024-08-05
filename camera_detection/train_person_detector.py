from ultralytics import YOLO
from urllib.parse import urlparse
import fiftyone as fo
import fiftyone.zoo as foz
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Function to load and filter dataset
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

# Function to train and validate YOLO models
def train_and_validate_models(models, data_yaml, epochs, imgsz, device, batch, project, datatest):
    for model_name in models:
        print("==========")
        print(f"Model: {model_name}")

        print("Model training...")
        model = YOLO(f"{model_name}.pt")
        train_result = model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, device=device, batch=batch, plots=True, seed=18, project=f"{project}/{model_name}")
        print("Train result: ", train_result)

        print("Model validations...")
        metrics = model.val(save_json=True)
        print("Metrics: ", metrics)
        print("Mean average precisions: ", metrics.box.maps)
        print("Testing predictions...")

        predictions = model.predict(source=datatest)

        for k, p in enumerate(predictions):
            url = datatest[k]
            parsed_url = urlparse(url=url)
            file_name = parsed_url.path.split('/')[-1]
            p.save(f"{project}/{model_name}/predicted/{file_name}")

# Main function to parse arguments and run the script
def main():
    parser = argparse.ArgumentParser(description="Load, filter, export COCO-2017 dataset, and train YOLO models.")
    parser.add_argument('--classes', type=str, nargs='+', default=['person'], help='Classes to filter in the dataset')
    parser.add_argument('--max_samples', type=int, default=10000, help='Maximum number of samples to load')
    parser.add_argument('--export_dir', type=str, default='./yolov5-coco-datasets', help='Directory to export the dataset')
    parser.add_argument('--models', type=str, nargs='+', default=['yolov8n', 'yolov8s', 'yolov8m'], help='List of YOLO models to train and validate')
    parser.add_argument('--data_yaml', type=str, default='./yolov5-coco-datasets/dataset.yaml', help='Path to the data YAML file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for training')
    parser.add_argument('--device', type=int, default=0, help='Device to use for training (0 for GPU, -1 for CPU)')
    parser.add_argument('--batch', type=int, default=8, help='Batch size for training')
    parser.add_argument('--project', type=str, default='./training', help='Project directory for training results')
    parser.add_argument('--datatest', type=str, nargs='+', default=[
        'https://cdn.antaranews.com/cache/1200x800/2023/10/13/Pengendara-Sepeda-Motor-Trotoar-060323-aaa-5.jpg',
        'https://img.harianjogja.com/posts/2022/11/14/1117643/jalur-pedestrian-malioboro.jpg',
        'https://assets.bwbx.io/images/users/iqjWHBFdfxIU/i.KTm08H6tuM/v1/1200x810.jpg',
        'https://static.promediateknologi.id/crop/0x0:0x0/0x0/webp/photo/radarjogja/2023/01/web-JOG-Pedestrian-Harus-Sesuai-Fungsinya-FAT-010122.jpg'
    ], help='List of URLs for testing predictions')

    args = parser.parse_args()

    datasets = load_and_filter_dataset(classes=args.classes, max_samples=args.max_samples)
    export_splits(datasets, export_dir=args.export_dir, classes=args.classes)
    train_and_validate_models(models=args.models, data_yaml=args.data_yaml, epochs=args.epochs, imgsz=args.imgsz, device=args.device, batch=args.batch, project=args.project, datatest=args.datatest)

if __name__ == "__main__":
    main()
