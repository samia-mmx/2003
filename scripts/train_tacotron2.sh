CUDA_VISIBLE_DEVICES=5 python train.py --load-mel-from-disk -o logs --init-lr 1e-3 --final-lr 1e-5 --epochs 200 -bs 32 --weight-decay 1e-6 --log-file nvlog.json --dataset-path training_data --training-anchor-dirs LJSpeech-1.1
