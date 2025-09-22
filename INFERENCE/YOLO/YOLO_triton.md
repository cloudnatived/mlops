

pip install -U ultralytics





trtexec --onnx=./yolov13x.onnx   --saveEngine=./model.trt   --fp16 --workspace=8192 --optShapes=images:1x3x640x640   --minShapes=images:1x3x640x640 --maxShapes=images:8x3x640x640


pip3 install tensorrt==8.6.1
pip3 install serialize
