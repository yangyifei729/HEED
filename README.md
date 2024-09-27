# Source Codes and Dataset for HEED submission
## Dataset

- We provide the "toy" train, dev data in `train_dev_data` such as `train_dev_data/V1_train_image_features_tsv_toy.tsv` and `train_dev_data/V1V2V3_dev_conll_MainImage_features_tsv_toy.tsv`, where the original text has been encoded by the XLM-RoBERTa tokenizer.
- The "toy" test set is in `test_data`.
- The "toy" original data is in `train_dev_data/original_toy.tsv`.



## Model Training and Evaluation
### Dependencies
- python >= 3.7
- torch >= 1.8.0 
- apex = 0.1
- tqdm
- numpy
- scikit-learn

### Description
- The code for training and testing on dev set is in `run_classifier_mt_moe.py`.
- The code for testing on the test set can be refered to `test_mt_moe.ipynb`.

### Model Training
You can train a model like in `ddp_train_mt_moe.sh`:
```bash
python -m torch.distributed.launch --nproc_per_node 8 run_classifier_mt_moe.py \
    --model_type xlmroberta \
    --model_name_or_path MODEL_PATH \
    --encoder_config_name ./data/config-xlmr-L3.json \
    --data_dir ./train_dev_data/ \
    --mt_training_config_path ./data/mt_config.cfg \
    --max_seq_length 512 \
    --per_gpu_train_batch_size 12 \
    --fp16 \
    --experts_per_modal 6 \
    --per_gpu_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 5 \
    --max_steps -1 \
    --learning_rate 1e-5 \
    --save_steps 2000 \
    --output_dir ./saved/test/ \
    --logging_dir ./saved/test/logs \
    --old_currency=USD,CNY,CHF,JPY,YEN,KRW,UAH,EUR,PLN,Ft \
    --new_currency=CNY,CHF,JPY,YEN,KRW,RON,Ft,EUR,PLN,UAH \
    --ratio_currency=0.05 \
    --currency=./data/currency.npy \
    --data_augmentation  \
    --num_hidden_layers 12 \
    --dev_file V1V2V3_dev_conll_MainImage_features_tsv_toy.tsv,V1V2V3_dev_conll_Name_features_tsv_toy.tsv,V1V2V3_dev_conll_Price_features_tsv_toy.tsv \
    --do_train \
    --do_eval \
    --eval_all_checkpoints \
    --hidden_size 768 \
    --output_hidden_states \
    --intermediate_size 3072 \
    --num_attention_heads 12 \
    --temperature_ratio=1.0 \
    --weight_decay 0.005 \
    --passage_index=1 \
    --overwrite_output_dir \
    --query_index 0 \
    --label_index 1 \
    --visual_feature_index 2 \
    --lazy_load_block_size 10000 \
    --lazy_load_block_size_for_multi_training_data 10000 \
    --logging_steps 1000 \
    --num_workers 4
```
### Model Test
You can test the trained checkpoints on the dev set like in `eval_mt_moe.sh`:
```bash
python run_classifier_mt_moe.py \
    --model_type xlmroberta \
    --model_name_or_path MODEL_PATH \
    --encoder_config_name ./data/config-xlmr-L3.json \
    --data_dir ./train_dev_data/ \
    --mt_training_config_path ./data/mt_config.cfg \
    --max_seq_length 512 \
    --per_gpu_train_batch_size 16 \
    --per_gpu_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --max_steps -1 \
    --learning_rate 1e-5 \
    --save_steps 363 \
    --output_dir ./saved/test/ \
    --logging_dir ./saved/test/ \
    --old_currency=USD,CNY,CHF,JPY,YEN,KRW,UAH,EUR,PLN,Ft \
    --new_currency=CNY,CHF,JPY,YEN,KRW,RON,Ft,EUR,PLN,UAH \
    --ratio_currency=0.05 \
    --currency=./data/currency.npy \
    --data_augmentation  \
    --num_hidden_layers 12 \
    --dev_file V1V2V3_dev_conll_MainImage_features_tsv_toy.tsv,V1V2V3_dev_conll_Name_features_tsv_toy.tsv,V1V2V3_dev_conll_Price_features_tsv_toy.tsv \
    --do_eval \
    --eval_all_checkpoints \
    --hidden_size 768 \
    --output_hidden_states \
    --intermediate_size 3072 \
    --num_attention_heads 12 \
    --temperature_ratio=1.0 \
    --weight_decay 0.005 \
    --passage_index=1 \
    --overwrite_output_dir \
    --query_index 0 \
    --label_index 1 \
    --visual_feature_index 2 \
    --lazy_load_block_size 100000 \
    --lazy_load_block_size_for_multi_training_data 100000 \
    --logging_steps 1000 \
    --num_workers 8
```

You can test the best checkpoint on the test set as shown in `test_mt_moe.ipynb`.