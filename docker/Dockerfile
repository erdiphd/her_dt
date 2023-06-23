
FROM nvidia/cuda:11.4.0-base-ubuntu20.04 

RUN apt-get update && apt-get install --no-install-recommends -y 
    

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  locales \
  git \
  libgl1-mesa-dri \
  menu \
  python \
  pip \
  mesa-utils \
  nano \
  net-tools \
  openbox \
  openssh-server \
  sudo \
  supervisor \
  terminator \
  tint2 \
  vim \
  x11-xserver-utils \
  x11vnc \
  xinit \
  xserver-xorg-video-dummy \
  xserver-xorg-input-void \
  websockify && \
  pip install supervisor-stdout && \
  apt-get -y clean

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    software-properties-common \
    net-tools \
    unzip \
    vim \
    virtualenv \
    wget \
    xpra \
    mpich \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


RUN locale-gen en_US.UTF-8 en_GB.UTF-8 de_DE.UTF-8 de_DE@euro
RUN echo 'LANG=en_US.UTF-8' >> /etc/locale.conf
# RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/user

# RUN adduser --disabled-password \
# --gecos '' user
# RUN adduser user sudo

RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> \
/etc/sudoers


# Create user
RUN useradd -s /bin/bash -d /home/user/ -m -G sudo user


RUN sudo mkdir -p /home/user/.mujoco \
    && wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -zxf mujoco.tar.gz --directory /home/user/.mujoco \
    && rm mujoco.tar.gz


# Mini Anaconda Installation

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -u -p /home/user/conda \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

USER user

ENV LD_LIBRARY_PATH /home/user/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV LD_PRELOAD /usr/lib/x86_64-linux-gnu/libGLEW.so
ENV MUJOCO_GL "glfw"

COPY environment.yml /home/user/environment.yml

RUN sudo chown -R user:user /home/user/environment.yml

RUN /bin/bash -c "source /home/user/conda/bin/activate && \
                  cd /home/user/ && \
                  conda env create -f environment.yml"



VOLUME [ "/home/user/her_dt/" ]     


ENV tag "her_result1"
ENV learn "normal"
ENV env "FetchPickAndPlace-v1"
ENV goal "interval"
ENV epochs "20"
ENV cycles "20"
ENV episodes "50"

COPY docker/startup.sh /home/user/
ENV PATH /home/user/.conda/envs/her_dt/bin:$PATH
CMD ["bin/bash", "home/user/startup.sh"]
