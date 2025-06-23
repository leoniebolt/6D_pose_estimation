# 6D_pose_estimation
Estimating the 6D pose of morobot parts with trained yolo model and megapose6D.

# DISCLAIMER
I got help from a collegue, since many things did not work with me.

Therefore, they may be some logic similiarities.

Also the .ply files are from them, since it did not work with mine, but with theirs.
I could, unfortunately not resolve this issue.
So therefore the box dimensions are the same.

# 1. Working environment
All the current packages are listed in the file environment_packages.txt

Python version: 3.9.23

My PyTorch Version: 1.11.0.post202

Cuda Version: 11.2



# 2. Requirements
The following things are required for this project to work:

Ultralytics
opencv-python
numpy
scipy
trimesh

In case you don't have these installed:

```bash
pip install ultralytics opencv-python numpy scipy trimesh

```

# 3. Installation
First download my GitHub repository:

```bash
git clone https://github.com/leoniebolt/6D_pose_estimation.git
cd 6D_pose_estimation
```

## megapose6d
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

In the folder morobot you should delete all the .txt files (in all folders and subfolders), otherwise an error occurs.

I somehow cannot make an empty folder in github.

The folder "morobot" contains the following:

- file camera_data.json
- file camera_intrinsic.json
- folders with a .txt file which should be deleted.

I have attached the meshes folder with .ply files into the submission. That folder shall be moved into the morobot folder.
After removing the .txt files and copying the meshes folder into the morobot folder, move it to the following two destinations:

6D_pose_estimation/data

6D_pose_estimation/megapose6d/local_data/examples



# 3 Starting

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


# 4. Explanation of Pipeline (main.py)
## 4.1 Object Detection with YOLO

- Loads a pretrained YOLOv8 model.
- Detects objects in RGB images
- Saves bounding boxes and class labels to JSON files
- Creates annotated detection images


![1_yolo_detected](https://github.com/user-attachments/assets/76f96cb0-619e-4ad6-a65f-90c991c712cc)




## 4.2 Preparation for MegaPose

- Copies RGB, depth, detection, camera intrinsics, mesh files into the megapose example directory.
- Validates 3D mesh files using trimesh. Since I had often problems with my .ply files, I implemented a debugging feature. I used the .ply files from a collegue, since for some reason, I couldn't export my own .ply files.


## 4.3 Pose Estimation with MegaPose

- Runs the official megapose inference script:
    - Detection visualization
    - Pose inference
    - Output visualization


![0_all_results](https://github.com/user-attachments/assets/ea2c35d2-7cf5-4e0e-af89-0086a7af951b)


 
## 4.4 Visualization

- Renders 3D bounding boxed and coordinate axes from predicted poses onte original image.
- Saves visualized output per frame for each image index.


![1_poses](https://github.com/user-attachments/assets/c219d9a4-0f54-4ca4-b650-47e18ada1fa1)



## 4.5 Postprocessing & Cleanup

- Saves results
- Deletes temporary intermediate files to keep workspace clean.
