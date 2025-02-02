python generate.py \
    --text "Ultra-realistic headshot of a person looking directly at the camera, centered composition, unobstructed face, professional studio lighting, realistic eyes, smooth skin texture, sharp focus, realistic proportions, smooth background blur, natural and detailed facial features, only head and shoulders in the frame, neutral expression, soft and even lighting, minimal shadows" \
    --negative_prompt "Full body, distorted proportions, blurry, cartoonish, side profiles, watermarks, exaggerated expressions, harsh lighting, cluttered background, multiple heads, unrealistic features, low resolution" \
    --input_image ./data/face_data/img_1.jpg \
    --optim_forward_guidance_wt 2.0 \
    --optim_num_steps 2 \
    --trials 10 \
    --ddim_steps 500 \
    --seed 0 \
    --optim_forward_guidance \
    --fr_crop \
    --optim_original_conditioning \
    --fr_model 2 \
    --optim_folder /scratch1/fxiao/noise/2_2_final \
    --ckpt sd-v1-5.ckpt \
