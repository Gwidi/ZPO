cd ../data
wget "http://images.cocodataset.org/zips/val2017.zip" -O val2017.zip
wget "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip" -O coco2017labels.zip
unzip val2017.zip
unzip coco2017labels.zip
rm val2017.zip
rm coco2017labels.zip
mkdir -p coco_val2017/images coco_val2017/labels
mv val2017/* coco_val2017/images/
mv coco/labels/val2017/* coco_val2017/labels/
rm -r val2017/ coco/
cd ../