python generate.py \
    --text "headshot of a woman" \
    --input_image ./data/face_data/img_0.png \
    --optim_forward_guidance_wt 1.0 \
    --optim_num_steps 2 \
    --ddim_steps 500 \
    --seed 234 \
    --optim_forward_guidance \
    --fr_crop \
    --optim_original_conditioning \
    --optim_folder ./exps/trial1 \
    --ckpt sd-v1-4.ckpt \
