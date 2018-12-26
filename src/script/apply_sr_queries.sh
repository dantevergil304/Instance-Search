cd ../../3rd_party/Image-Super-Resolution
for f in ../../data/processed_data/faces/queries/detect_before_mask/*.bmp; do
	CUDA_VISIBLE_DEVICES=2, python main.py "$f" --save_intermediate=false --model="distilled_rnsr"
done
cd ../../src/script/
