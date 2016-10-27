#!/bin/bash

function md5_check {
    if [ -f $1 ]; then
      echo "File $1 already exists. Checking md5..."
      os=`uname -s`
      if [ "$os" = "Linux" ]; then
        checksum=`md5sum $1 | awk '{ print $1 }'`
      elif [ "$os" = "Darwin" ]; then
        checksum=`cat $1 | md5`
      fi
      if [ "$checksum" = "$2" ]; then
        echo "Checksum is correct. No need to download."
        CHECK_CORRECT=1
      else
        echo "Checksum is incorrect. Need to download again."
        echo $checksum
        echo $2
        CHECK_CORRECT=0
      fi
    fi
}

function download_if_needed {
    CHECK_CORRECT=0
    md5_check $2 $3
    if [ "$CHECK_CORRECT" = "0" ]; then
        wget $1 -O $2
    fi
}
ROOT_URL=http://geometry.cs.ucl.ac.uk/mhueting/proj/scenemap/data/scenemap/data

cd data

# rgb
echo '-----------------------------'
echo 'Downloading RGB input data...'
echo '-----------------------------'
CHECKSUM=b198d8d5fb0dca17fa9ff858640b203d
URL=$ROOT_URL/rgb.bin
download_if_needed $URL rgb.bin $CHECKSUM

# depth
echo '-------------------------'
echo 'Downloading depth data...'
echo '-------------------------'
CHECKSUM=eccdfb562f4a62217c448c01ea46cc18
URL=$ROOT_URL/dep.bin
download_if_needed $URL dep.bin $CHECKSUM

# semantics
echo '-----------------------------------------'
echo 'Downloading semantic segmentation data...'
echo '-----------------------------------------'
CHECKSUM=f2dd480eae88bb62775a4cb183de5f28
URL=$ROOT_URL/sem.bin
download_if_needed $URL sem.bin $CHECKSUM

# map
echo '----------------------------------------'
echo 'Downloading scene map estimation data...'
echo '----------------------------------------'
CHECKSUM=bc1772608e3bc8dc3a678ce3827e1339
URL=$ROOT_URL/map.bin
download_if_needed $URL map.bin $CHECKSUM

echo '-----------------------------------------------------------------'
echo 'All done \o/ Run this file again to check MD5 checksums of files.'
echo '-----------------------------------------------------------------'
