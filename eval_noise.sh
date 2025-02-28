conda activate dmsr
for i in $(seq 0 18);
do
    python evaluate.py --data real_test --model ./pretrained/real_model.pth --corruption $i 
done