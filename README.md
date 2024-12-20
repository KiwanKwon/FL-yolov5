# YOLOv5 Training and Federated Learning

## Benchmark Dataset Download
[Download Benchmark Dataset](https://drive.google.com/file/d/1Ec031OKDUfvuDW_WvF5vqmIJATb4wJlI/view?usp=drivesdk)

## Installation Instructions

1. Clone the YOLOv5 repository:
   ```bash
   git clone https://github.com/ultralytics/yolov5

2. Navigate to the YOLOv5 directory and create a new Conda environment:
   ```bash
   cd yolov5
   conda create --name yolo python=3.8

4. Activate the Conda environment:
   ```bash
   conda activate yolo

6. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

## Training
### Centralized Learning (CL)
1. To train the model with centralized learning:
   ```bash
   python train.py --data /path/to/your/datasets/datasets.yaml --weights yolov5s.pt --img 640 --batch 16 --epochs 10 --name cl
   #Replace /path/to/your/datasets/datasets.yaml with the actual path to your dataset configuration file.

### Federated Learning (FL)
2. To run federated learning:
   ```bash
   python global_server.py


