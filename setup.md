WSL2 上で構築
(base) miu@garnet:/mnt/c/MMD/EDGE$

## cuda 11.6 のインストール

```
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda-11-6
```

```
Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
Some packages could not be installed. This may mean that you have
requested an impossible situation or if you are using the unstable
distribution that some required packages have not yet been created
or been moved out of Incoming.
The following information may help to resolve the situation:

The following packages have unmet dependencies:
 libcufile-11-6 : Depends: liburcu6 but it is not installable
E: Unable to correct problems, you have held broken packages.
```

```
(base) miu@garnet:~/cuda11.6$ wget http://ftp.jp.debian.org/debian/pool/main/libu/liburcu/liburcu6_0.12.2-1_amd64.deb
(base) miu@garnet:~/cuda11.6$ sudo dpkg -i liburcu6_0.12.2-1_amd64.deb
```

```
(base) miu@garnet:~/cuda11.6$ sudo dpkg -i liburcu6_0.12.2-1_amd64.deb
Selecting previously unselected package liburcu6:amd64.
(Reading database ... 59731 files and directories currently installed.)
Preparing to unpack liburcu6_0.12.2-1_amd64.deb ...
Unpacking liburcu6:amd64 (0.12.2-1) ...
Setting up liburcu6:amd64 (0.12.2-1) ...
Processing triggers for libc-bin (2.35-0ubuntu3.5) ...
/sbin/ldconfig.real: /usr/lib/wsl/lib/libcuda.so.1 is not a symbolic link
```

```
(base) miu@garnet:/usr/lib/wsl/lib$ sudo rm libcuda.so
(base) miu@garnet:/usr/lib/wsl/lib$ sudo rm libcuda.so.1
(base) miu@garnet:/usr/lib/wsl/lib$ sudo ln -s libcuda.so.1.1 libcuda.so
(base) miu@garnet:/usr/lib/wsl/lib$ sudo ln -s libcuda.so.1.1 libcuda.so.1
```

```
(base) miu@garnet:~/cuda11.6$ sudo dpkg -i liburcu6_0.12.2-1_amd64.deb
```

## 初期環境

```
conda create -n edge python=3.9
conda activate edge
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
```

## pytorch3d

```
conda install pytorch3d -c pytorch3d
```

## jukemirlib

```
pip install git+https://github.com/rodrigo-castellon/jukemirlib.git
```

## accelerate

```
pip install accelerate
```

## wine

```
sudo dpkg --add-architecture i386
sudo mkdir -pm755 /etc/apt/keyrings
sudo wget -O /etc/apt/keyrings/winehq-archive.key https://dl.winehq.org/wine-builds/winehq.key
sudo wget -NP /etc/apt/sources.list.d/ https://dl.winehq.org/wine-builds/ubuntu/dists/jammy/winehq-jammy.sources
sudo apt update
sudo apt install --install-recommends winehq-stable
```

## 学習モデルダウンロード

```
(edge) miu@garnet:/mnt/c/MMD/EDGE$ chmod 744 download_model.sh
(edge) miu@garnet:/mnt/c/MMD/EDGE$ ./download_model.sh
```

## その他

```
pip install wandb
pip install matplotlib
pip install einops
pip install p-tqdm
```

```
python test.py --music_dir custom_music/  --save_motions
```

## ffmpeg のセッティング

```
# FFmpegコンパイルに必要なものを導入
sudo apt-get update -qq && sudo apt-get -y install \
  autoconf \
  automake \
  build-essential \
  cmake \
  git-core \
  libass-dev \
  libfreetype6-dev \
  libgnutls28-dev \
  libunistring-dev \
  libmp3lame-dev \
  libtool \
  libvorbis-dev \
  pkg-config \
  texinfo \
  wget \
  yasm \
  zlib1g-dev

# 作業ディレクトリ作成
mkdir -p ~/ffmpeg_sources ~/bin

# NASM インストール
sudo apt-get -y　install nasm

# libx264 インストール
sudo apt-get -y install libx264-dev

# libx265 インストール
sudo apt-get -y install libx265-dev libnuma-dev

# libvpx インストール
sudo apt-get -y install libvpx-dev

# libfdk-aac インストール
sudo apt-get -y install libfdk-aac-dev

# libopus インストール
sudo apt-get -y install libopus-dev

# libaom インストール
sudo apt-get -y install libaom-dev

# NVIDIA codec API インストール
cd ~/ffmpeg_sources && \
git -C nv-codec-headers pull 2> /dev/null || git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers && \
cd nv-codec-headers && \
make && \
make install PREFIX="$HOME/ffmpeg_build"

# FFmpeg インストール

cd ~/ffmpeg_sources && \
wget -O ffmpeg-snapshot.tar.bz2 https://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2 && \
tar xjvf ffmpeg-snapshot.tar.bz2 && \
cd ffmpeg && \
PATH="$HOME/bin:$PATH" PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig" ./configure \
  --prefix="$HOME/ffmpeg_build" \
  --pkg-config-flags="--static" \
  --extra-cflags="-I$HOME/ffmpeg_build/include" \
  --extra-ldflags="-L$HOME/ffmpeg_build/lib" \
  --enable-cuda-nvcc \
  --nvccflags="-gencode arch=compute_52,code=sm_52 -O2" \
  --enable-cuvid \
  --enable-nvenc \
  --enable-libnpp \
  --extra-cflags="-I/usr/local/cuda-12.2/include" \
  --extra-ldflags="-L/usr/local/cuda-12.2/lib64" \
  --extra-libs="-lpthread -lm" \
  --ld="g++" \
  --bindir="$HOME/bin" \
  --enable-gpl \
  --enable-gnutls \
  --enable-libaom \
  --enable-libass \
  --enable-libfdk-aac \
  --enable-libfreetype \
  --enable-libmp3lame \
  --enable-libopus \
  --enable-libvorbis \
  --enable-libvpx \
  --enable-libx264 \
  --enable-libx265 \
  --enable-nonfree && \
PATH="$HOME/bin:$PATH" make && \
make install && \
hash -r

# パスを通したり
source ~/.profile
echo "MANPATH_MAP $HOME/bin $HOME/ffmpeg_build/share/man" >> ~/.manpath
```

python test.py --music_dir custom_music/02/  --save_motions
