from ultralytics import YOLO
from urllib.parse import urlparse
import fiftyone as fo
import fiftyone.zoo as foz
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

    # Check if the device is available
    if args.device == '0' and torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print(f"Device: {device}")

    train_and_validate_models(models=args.models, data_yaml=args.data_yaml, epochs=args.epochs, imgsz=args.imgsz, device=args.device, batch=args.batch, project=args.project, datatest=args.datatest)

if __name__ == "__main__":
    main()
