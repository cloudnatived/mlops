## Windows in Docker Container

```
windowhttps://zhuanlan.zhihu.com/p/686351917  # 把 Windows 装进 Docker 容器里

git clone https://github.com/dockur/windows.git
cd windows
nerdctl build -t dockurr/windows .
docker build -t dockurr/windows .
#需要启动 buildkit.service

docker compose up
nerdctl compose up

------------------------------------ cat compose.yml
services:
  windows:
    image: hub.rat.dev/dockurr/windows
    container_name: windows-10  # windows10
    environment:
      VERSION: "10"   # windows10
    devices:
      - /dev/kvm
      - /dev/net/tun
    cap_add:
      - NET_ADMIN
    ports:
      - 8006:8006
      - 3389:3389/tcp
      - 3389:3389/udp
    stop_grace_period: 2m
------------------------------------



nerdctl container ls -a |grep windows
docker container ls -a |grep windows

Value	Version	Size
11	Windows 11 Pro	5.4 GB
11l	Windows 11 LTSC	4.7 GB
11e	Windows 11 Enterprise	4.0 GB
10	Windows 10 Pro	5.7 GB
10l	Windows 10 LTSC	4.6 GB
10e	Windows 10 Enterprise	5.2 GB
8e	Windows 8.1 Enterprise	3.7 GB
7e	Windows 7 Enterprise	3.0 GB
ve	Windows Vista Enterprise	3.0 GB
xp	Windows XP Professional	0.6 GB

------------------------------------    compose.yml
services:
  windows:
    image: dockurr/windows
    container_name: windows-10 # windows10
    environment:
      VERSION: "10"   # windows10
    devices:
      - /dev/kvm
      - /dev/net/tun
    cap_add:
      - NET_ADMIN
    ports:
      - 8006:8006
      - 3389:3389/tcp
      - 3389:3389/udp
    restart: always
    stop_grace_period: 2m
------------------------------------


cat Dockerfile
------------------------------------
ARG VERSION_ARG="latest"
FROM scratch AS build-amd64

COPY --from=hub.rat.dev/qemux/qemu:7.12 / /

ARG DEBCONF_NOWARNINGS="yes"
ARG DEBIAN_FRONTEND="noninteractive"
ARG DEBCONF_NONINTERACTIVE_SEEN="true"

RUN set -eu && \
    apt-get update && \
    apt-get --no-install-recommends -y install \
        samba \
        wimtools \
        dos2unix \
        cabextract \
        libxml2-utils \
        libarchive-tools \
        netcat-openbsd && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY --chmod=755 ./src /run/
COPY --chmod=755 ./assets /run/assets

ADD --chmod=755 https://raw.githubusercontent.com/christgau/wsdd/refs/tags/v0.9/src/wsdd.py /usr/sbin/wsdd
ADD --chmod=664 https://github.com/qemus/virtiso-whql/releases/download/v1.9.47-0/virtio-win-1.9.47.tar.xz /var/drivers.txz

FROM dockurr/windows-arm:${VERSION_ARG} AS build-arm64
FROM build-${TARGETARCH}

ARG VERSION_ARG="0.00"
RUN echo "$VERSION_ARG" > /run/version

VOLUME /storage
EXPOSE 3389 8006

ENV VERSION="10"
ENV RAM_SIZE="8G"
ENV CPU_CORES="4"
ENV DISK_SIZE="64G"

ENTRYPOINT ["/usr/bin/tini", "-s", "/run/entry.sh"]
------------------------------------


```

