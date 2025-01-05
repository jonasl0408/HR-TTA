DATASET="imagenet_c"     # cifar10_c cifar100_c imagenet_c
METHOD="tent"        # source eata eataE10 hrtta hrttaE10 tent tentE10 cotta cottaE10 sar sarE10

echo DATASET: $DATASET
echo METHOD: $METHOD

GPU_id=0

CUDA_VISIBLE_DEVICES="$GPU_id" python test-time.py --cfg best_cfgs/Online_TTA/${DATASET}/${METHOD}.yaml --output_dir "test-time-evaluation/${DATASET}/${METHOD}" &