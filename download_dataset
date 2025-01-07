#!/bin/bash

# 检查输入参数
FILE=$1

# 如果参数不在预定义的列表中，打印可用的数据集并退出
if [[ $FILE != "ae_photos" && $FILE != "apple2orange" && $FILE != "summer2winter_yosemite" &&  $FILE != "horse2zebra" && $FILE != "monet2photo" && $FILE != "cezanne2photo" && $FILE != "ukiyoe2photo" && $FILE != "vangogh2photo" && $FILE != "maps" && $FILE != "cityscapes" && $FILE != "facades" && $FILE != "iphone2dslr_flower" ]]; then
    echo "Available datasets are: apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, ae_photos"
    exit 1
fi

# 设置下载 URL 和文件路径
URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/$FILE.zip
ZIP_FILE=./datasets/$FILE.zip
TARGET_DIR=./datasets/$FILE

# 创建 datasets 目录（如果不存在）
mkdir -p ./datasets

# 下载数据集 zip 文件，如果下载失败则退出
if ! wget -N $URL -O $ZIP_FILE; then
    echo "Download failed!"
    exit 1
fi

# 解压 zip 文件到 datasets 目录，如果解压失败则退出
if ! unzip $ZIP_FILE -d ./datasets/; then
    echo "Unzip failed!"
    exit 1
fi

# 删除已下载的 zip 文件，节省空间
rm $ZIP_FILE

# 适配到项目所需的目录结构
mkdir -p "$TARGET_DIR/train" "$TARGET_DIR/test"

# 移动并重命名文件夹
mv "$TARGET_DIR/trainA" "$TARGET_DIR/train/A"
mv "$TARGET_DIR/trainB" "$TARGET_DIR/train/B"
mv "$TARGET_DIR/testA" "$TARGET_DIR/test/A"
mv "$TARGET_DIR/testB" "$TARGET_DIR/test/B"

# 提示完成
echo "Dataset $FILE downloaded and organized successfully!"
