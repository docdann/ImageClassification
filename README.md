## Install Dependencies

```
python -m venv venv

.\venv\Scripts\activate

python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

python -m pip install -r .\requirements.txt

python.exe -m pip install --upgrade pip
```

## PROTOBUF - Generate pb2 file
```
protoc --python_out=. .\image_caption.proto
```

## Docker

### Build and Run Docker


```bash
docker build -t image_classification_ws_api -f dockerfile_ws .

docker run --gpus all -it -p 8000:8000 image_classification_ws_api:latest
```




### Example Client

```bash
python .\image_classification_ws_client.py
```

