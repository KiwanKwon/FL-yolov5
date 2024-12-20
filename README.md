# DD-FL
benchmark dataset download:
https://drive.google.com/file/d/1Ec031OKDUfvuDW_WvF5vqmIJATb4wJlI/view?usp=drivesdk

git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
conda create --name yolo python=3.8
conda activate yolo
pip install -r requirements.txt  # install

python train.py --data /path/to/your/datasets/datasets.yaml --weights yolov5s.pt --img 640 --batch 16 --epochs 10 --name cl #centralized learning



