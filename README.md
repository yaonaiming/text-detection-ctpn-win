# text-detection-ctpn-win

This repo is a text detection project based on ctpn (connectionist text proposal network). It is originally implemented in tensorflow on Linux and can be found in [here](https://github.com/eragonruan/text-detection-ctpn). In this repo, I fixed all MSVC compilation issues by patching gpu_nms.pyx, gpu_nms.cpp, setup.py, and _msvccompiler.py, so that it can be run on Windows 10 with Python 3.6.

The origin repo in caffe can be found in [here](https://github.com/tianzhi0549/CTPN). Also, the origin paper can be found in [here](https://arxiv.org/abs/1609.03605). For more detail about the paper and code, see this [blog](http://slade-ruan.me/2017/10/22/text-detection-ctpn/).

***

# setup
- requirements: tensorflow1.3, cython0.24, opencv-python, easydict (recommend to install Anaconda)
- if you do not have a gpu device, follow here to [setup](https://github.com/eragonruan/text-detection-ctpn/issues/43)
- You can directly use lib/utils (i.e. bbox.pyd, cython_nms.pyd, and gpu_nms.pyd). No additional compilation step is needed unless errors occur. See the last section for more inforamtion.
***

# parameters
there are some parameters you may need to modify according to your requirement, you can find them in ctpn/text.yml
- USE_GPU_NMS # whether to use nms implemented in cuda or not
- DETECT_MODE # H represents horizontal mode, O represents oriented mode, default is H
- checkpoints_path # the model I provided is in checkpoints/, if you train the model by yourself,it will be saved in output/
***
# demo
- put your images in data/demo, the results will be saved in data/results, and run demo in the root 
```shell
python ./ctpn/demo.py
```
***
# training
## prepare data
- First, download the pre-trained model of VGG net and put it in data/pretrain/VGG_imagenet.npy. you can download it from [google drive](https://drive.google.com/open?id=0B_WmJoEtfQhDRl82b1dJTjB2ZGc) or [baidu yun](https://pan.baidu.com/s/1kUNTl1l). 
- Second, prepare the training data as referred in paper, or you can download the data I prepared from previous link. Or you can prepare your own data according to the following steps. 
- Modify the path and gt_path in prepare_training_data/split_label.py according to your dataset. And run
```shell
cd prepare_training_data
python split_label.py
```
- it will generate the prepared data in current folder, and then run
```shell
python ToVoc.py
```
- to convert the prepared training data into voc format. It will generate a folder named TEXTVOC. move this folder to data/ and then run
```shell
cd ../data
ln -s TEXTVOC VOCdevkit2007
```
## train 
Simplely run
```shell
python ./ctpn/train_net.py
```
- you can modify some hyper parameters in ctpn/text.yml, or just used the parameters I set.
- The model I provided in checkpoints is trained on GTX1070 for 50k iters.
- If you are using cuda nms, it takes about 0.2s per iter. So it will takes about 2.5 hours to finished 50k iterations.
***
# roadmap
- [x] cython nms
- [x] cuda nms
- [x] python2/python3 compatblity
- [x] tensorflow1.3
- [x] delete useless code
- [x] loss function as referred in paper
- [x] oriented text connector
- [x] BLSTM
- [ ] side refinement
***
# some results
`NOTICE:` all the photos used below are collected from the internet. If it affects you, please contact me to delete them.
<img src="/data/results/001.jpg" width=320 height=240 /><img src="/data/results/002.jpg" width=320 height=240 />
<img src="/data/results/003.jpg" width=320 height=240 /><img src="/data/results/004.jpg" width=320 height=240 />
<img src="/data/results/009.jpg" width=320 height=480 /><img src="/data/results/010.png" width=320 height=320 />
***
## oriented text connector
- oriented text connector has been implemented, i's working, but still need futher improvement.
- left figure is the result for DETECT_MODE H, right figure for DETECT_MODE O
<img src="/data/results/007.jpg" width=320 height=240 /><img src="/data/oriented_results/007.jpg" width=320 height=240 />
<img src="/data/results/008.jpg" width=320 height=480 /><img src="/data/oriented_results/008.jpg" width=320 height=480 />
***
# compiling lib/utils on Windows
- prerequisite: Win 10, Python 3.6, Visual Studio 2015 (MSVC toolchain is required), CUDA 8 (nvcc is required)
- requirements: cython 0.24, (recommend to install Anaconda)
- 1) open gpu_nms.pyx, go to line 25, replace
```c
cdef np.ndarray[np.int_t, ndim=1] \
```
as
```c
cdef np.ndarray[np.intp_t, ndim=1] \
```
- 2) open a cmd window and run
```shell
cd lib/utils
cython bbox.pyx
cython cython_nms.pyx
cython gpu_nms.pyx

set CUDAHOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0
set VS140COMNTOOLS=C:\Program Files (x86)\Microsoft Visual Studio 14.0\Common7\Tools\
set PATH=%PATH%;C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin
```
- 3) open gpu_nms_cpp, find _nms(...) and replace __pyx_t_5numpy_int32_t * as int*
```c
_nms((&(*__Pyx_BufPtrStrided1d(int*, __pyx_pybuffernd_keep.rcbuffer->pybuffer.buf, __pyx_t_10, __pyx_pybuffernd_keep.diminfo[0].strides))), (&__pyx_v_num_out), (&(*__Pyx_BufPtrStrided2d(__pyx_t_5numpy_float32_t *, __pyx_pybuffernd_sorted_dets.rcbuffer->pybuffer.buf, __pyx_t_12, __pyx_pybuffernd_sorted_dets.diminfo[0].strides, __pyx_t_13, __pyx_pybuffernd_sorted_dets.diminfo[1].strides))), __pyx_v_boxes_num, __pyx_v_boxes_dim, __pyx_t_14, __pyx_v_device_id);
```
- 4) backup <Anaconda root>\Lib\distutils\_msvccompiler.py, then replace 
```python
best_version, best_dir = _find_vc2017() 
``` 
as
```python
best_version, best_dir = _find_vc2017() 
best_version = None
```
go to line 411, insert
```python
            elif ext in ['.cu']:
                try:
                    args = [r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin\nvcc.exe'] + pp_opts + \
                           [r'-IC:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\include',
                            src,
                            '-odir', r'build\temp.win-amd64-3.6\Release',  # check this path if errors occur
                            '-arch=sm_35',
                            '--ptxas-options=-v',
                            '-c',
                            '-Xcompiler',
                            "/wd4819,/EHsc,/W3,/nologo,/O2,/Zi,/MD", 
                            '--compiler-options']
                    self.spawn(args)
                except DistutilsExecError as msg:
                    raise CompileError(msg)
                continue
```
- 5) use the fixed lib/utils/setup.py to build the library.
```shell
python setup.py build_ext --inplace
```
- 6) it works if bbox.pyd, cython_nms.pyd, and gpu_nms.pyd are built.
