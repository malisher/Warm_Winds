FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

SHELL ["/bin/bash", "-c"]
ENV NVIDIA_DRIVER_CAPABILITIES all

ENV TZ=Asia/Almaty
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN DEBIAN_FRONTEND="noninteractive" apt-get update && apt-get -y install tzdata

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip git wget build-essential ssh gcc sudo \
        g++ \
        gdb \
        clang \
        cmake \
        rsync \
        tar \
        ffmpeg libsm6 libxext6 \
        python3-pip \
        libsm6 \
        libxext6 \
        libxrender-dev \
        python3 \
        python3-dev \
        python3-pip \
        git \
        wget \
        build-essential \
        ssh \
        gcc \
        sudo \
        g++ \
        gdb \
        clang \
        cmake \
        rsync \
        tar \
        ffmpeg \
        libsm6 \
        libxext6 \
        python3-dev \
        && apt-get clean

RUN /opt/conda/bin/pip install --upgrade pip
RUN /opt/conda/bin/pip3 install --upgrade pip
# RUN /opt/conda/bin/pip install --no-cache-dir setuptools && /opt/conda/bin/pip3 install --no-cache-dir scikit-build opencv-python numpy scipy imgaug scikit-image scikit-learn albumentations jupyterlab tqdm matplotlib pandas catalyst augmentation
RUN /opt/conda/bin/pip install --no-cache-dir setuptools && /opt/conda/bin/pip3 install --no-cache-dir albumentations coremltools ipython matplotlib numpy onnx onnxruntime onnxsim opencv_python openpyxl pandas Pillow psutil PyYAML Requests scipy seaborn torch torchvision tqdm

# Setting some environment variables.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib" \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN ( \
    echo 'LogLevel DEBUG2'; \
    echo 'PermitRootLogin yes'; \
    echo 'PasswordAuthentication yes'; \
    echo 'Subsystem sftp /usr/lib/openssh/sftp-server'; \
  ) > /etc/ssh/sshd_config_test_clion \
  && mkdir /run/sshd

RUN useradd -m user \
  && yes password | passwd user && adduser user sudo
#USER user

RUN usermod -s /bin/bash user

CMD ["/usr/sbin/sshd", "-D", "-e", "-f", "/etc/ssh/sshd_config_test_clion"]
