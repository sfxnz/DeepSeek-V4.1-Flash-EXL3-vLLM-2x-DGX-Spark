#!/bin/bash
# In-container (entrypoint bash, so no python and no sitecustomize runs; CPU only): dump what
# compare_images.py compares between two images. Writes /out/<TAG>.*:
#   files.sha256  sha256 of every file under the python trees the serve imports from
#   distinfo.txt  installed dist-info names (package versions)
#   so.sha256     vllm_exl3_c extension hash
#   elf/          its embedded sm_121a cubins (cuobjdump -xelf all; the image has no nvdisasm for -sass)
#   patch.sha256  /opt/dsv41-patch (baked patch dir);  site.sha256  baked sitecustomize
set -eu
TAG=$1
SP=/usr/local/lib/python3.12/dist-packages
cd "$SP"
find vllm vllm_exl3 flashinfer exllamav3 -type f ! -name '*.pyc' ! -path '*/__pycache__/*' -print0 \
  | sort -z | xargs -0 sha256sum > "/out/$TAG.files.sha256"
ls -d *.dist-info | sort > "/out/$TAG.distinfo.txt"
SO=$(ls "$SP"/vllm_exl3_c*.so)
sha256sum "$SO" > "/out/$TAG.so.sha256"
mkdir -p "/out/$TAG.elf" && (cd "/out/$TAG.elf" && /usr/local/cuda/bin/cuobjdump -xelf all "$SO" > /dev/null)
(cd /opt/dsv41-patch && find . -type f ! -name '*.pyc' ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum) \
  > "/out/$TAG.patch.sha256"
sha256sum /usr/lib/python3.12/sitecustomize.py > "/out/$TAG.site.sha256"
chown -R "${HOST_UID:-1000}:${HOST_GID:-1000}" /out
