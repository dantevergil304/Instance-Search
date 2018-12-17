cd ../../Image-Super-Resolution
for f in ../data/processed_data/faces/queries/*.bmp; do
	CUDA_VISIBLE_DEVICES=2, python main.py "$f" --save_intermediate=false
done
cd ../src/script/
