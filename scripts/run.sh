python train.py \
    --environment mobile\
    --encoder_type pixel \
    --action_repeat 1 \
    --save_tb --pre_transform_image_size 100 --image_size 84 \
    --work_dir ./log \
    --agent curl_sac --frame_stack 3 \
    --seed 1 --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 10000 \
    --batch_size 256 --pre_training_steps 10000 --num_train_steps 300000
