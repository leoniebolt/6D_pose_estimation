## 6D_pose_estimation
Estimating the 6D pose of morobot parts with trained yolo model and megapose6D.

## Working environment
All the current packages are listed in the file environment_packages.txt
Python version: 3.9.23
My PyTorch Version: 1.11.0.post202
Cuda Version: 11.2


## 2. Requirements
The following things are required for this project to work:
Ultralytics
Conda
PyTorch

```bash
pip install ultralytics
```

## 3. Installation
First download my GitHub repository:
```bash
git clone https://github.com/leoniebolt/6D_pose_estimation.git
cd 6D_pose_estimation
```

# megapose6d
The GitHub of megapose6d: https://github.com/megapose6d/megapose6d/tree/master

```bash
git clone https://github.com/megapose6d/megapose6d.git
cd megapose6d && git submodule update --init
```

BEFORE creating the conda environment, you have to make two changes:
In the megapose6d/conda/environment_full.yaml file:
The notebook version should be notebook version should be set to 6.4.12 [line 19]:
```bash
notebook=6.4.12
```
To be able to use rgb data, change in the file megpose6d/src/megapose/scripts/run_inference_on_example.py the following (simply add "-icp") [line 213]:
```bash
    parser.add_argument("--model", type=str, default="megapose-1.0-RGB-multi-hypothesis-icp")
```

Then proceed with the installation:
```bash
conda env create -f conda/environment_full.yaml
conda activate megapose
pip install -e .
python -m megapose.scripts.download --megapose_models
```

Then either download the demo data, or simply use the morobot data.

If you choose to use the morobot data right away, you have to make a folder.
For that go into the folder local_data and create the directory "examples":

```
cd local_data
mkdir examples
```

Into that folder move the morobot folder in 6D_pose_estimation, which contains the following:
- Folder "meshes":
  In there you should put the .ply files. Every morobot part into its respective folder in "meshes".
  So for example for the morobot-s_Achse-1A_gray: meshes/1A_gray/1A_gray.ply, and so on.
- file camera_data.json
- file camera_intrinsic.json


## 3 Starting
While being in the 6D_pose_estimation folder and activated conda environment run:
```bash
python main.py
```

In case you get the following error (like I did multiple times):
RuntimeError: CUDA out of memory. Tried to allocate 600.00 MiB (GPU 0; 5.77 GiB total capacity; 2.46 GiB already allocated; 539.44 MiB free; 3.19 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF

simply put the following into your terminal:
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

and run the main.py file again.

## 4. Explanation
