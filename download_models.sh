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
MODEL_FILE=best_model.t7
OPTIM_FILE=best_optim_state.t7
LAT_FILE=latest.t7
ROOT_URL=http://geometry.cs.ucl.ac.uk/mhueting/proj/scenemap/data/scenemap/models

cd models

# depth
echo '-------------------------------------'
echo 'Downloading depth estimation model...'
echo '-------------------------------------'
cd depth
MODEL_CHECKSUM=099fa3487455eb6efca862c42324dc5d
MODEL_URL=$ROOT_URL/depth/$MODEL_FILE
download_if_needed $MODEL_URL $MODEL_FILE $MODEL_CHECKSUM
OPTIM_CHECKSUM=eaf35674dd788393e005c8dd351ea693
OPTIM_URL=$ROOT_URL/depth/$OPTIM_FILE
download_if_needed $OPTIM_URL $OPTIM_FILE $OPTIM_CHECKSUM
LAT_CHECKSUM=eb967849fab5a6f16a555dac570ffc45
LAT_URL=$ROOT_URL/depth/$LAT_FILE
download_if_needed $LAT_URL $LAT_FILE $LAT_CHECKSUM
cd ..

# semantics
echo '-------------------------------------'
echo 'Downloading semantic segmentation model...'
echo '-------------------------------------'
cd semantics
MODEL_CHECKSUM=ac12a0fbf5232949074ae6e41576fa26
MODEL_URL=$ROOT_URL/semantics/$MODEL_FILE
download_if_needed $MODEL_URL $MODEL_FILE $MODEL_CHECKSUM
OPTIM_CHECKSUM=a58ec6afef153c56d0b902e252c13a2a
OPTIM_URL=$ROOT_URL/semantics/$OPTIM_FILE
download_if_needed $OPTIM_URL $OPTIM_FILE $OPTIM_CHECKSUM
LAT_CHECKSUM=a276b39ab1740d87092d58a1168e047a
LAT_URL=$ROOT_URL/semantics/$LAT_FILE
download_if_needed $LAT_URL $LAT_FILE $LAT_CHECKSUM
cd ..

# map
echo '-------------------------------------'
echo 'Downloading scene map estimation model...'
echo '-------------------------------------'
cd map
MODEL_CHECKSUM=6808f72610daeed503b23e62fcb19329
MODEL_URL=$ROOT_URL/map/best_model_16x16.t7
download_if_needed $MODEL_URL best_model_16x16.t7 $MODEL_CHECKSUM
OPTIM_CHECKSUM=84c85a3a818f92745d444ddcd02fa882
OPTIM_URL=$ROOT_URL/map/best_optim_state_16x16.t7
download_if_needed $OPTIM_URL best_optim_state_16x16.t7 $OPTIM_CHECKSUM
LAT_CHECKSUM=fb64701003e7a16b2917dfadb6c3d31f
LAT_URL=$ROOT_URL/map/$LAT_FILE
download_if_needed $LAT_URL $LAT_FILE $LAT_CHECKSUM
cd ..

echo '-----------------------------------------------------------------'
echo 'All done \o/ Run this file again to check MD5 checksums of files.'
echo '-----------------------------------------------------------------'
