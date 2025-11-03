python test.py \
--content_path /home/admin/AdaAttN/datasets/biocop_files/biocop_512 \
--style_path /home/admin/AdaAttN/datasets/biocop_files/biocop_512 \
--name AdaAttN_second_attn_try \
--model adaattnattn \
--dataset_mode unaligned \
--load_size 256 \
--crop_size 256 \
--image_encoder_path /home/admin/AdaAttN/checkpoints/vgg_normalised.pth \
--gpu_ids 0 \
--shallow_layer \
--num_test 30 \
--skip_connection_3 \

# CHECK CROPPING