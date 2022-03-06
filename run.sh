cd wrap
./compile.sh
cd ..
python train.py --database sift --method angular --dout 32 --epochs 4 --batch_size 512 --save 1 --save_knn_1k 1 --val_freq_search 0