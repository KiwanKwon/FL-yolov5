# YOLOv5 Training and Federated Learning

## Benchmark Dataset Download
[Download Benchmark Dataset](https://drive.google.com/file/d/1Ec031OKDUfvuDW_WvF5vqmIJATb4wJlI/view?usp=drivesdk)

## Installation Instructions

1. Clone the YOLOv5 repository:
   ```bash
   git clone https://github.com/ultralytics/yolov5

2. Navigate to the YOLOv5 directory and create a new Conda environment:
   cd yolov5
   conda create --name yolo python=3.8

3. Activate the Conda environment:
   conda activate yolo

4. Install the required dependencies:
   pip install -r requirements.txt

5. python train.py --data /path/to/your/datasets/datasets.yaml --weights yolov5s.pt --img 640 --batch 16 --epochs 10 --name cl



