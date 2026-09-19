### To start preprocessing:

```bash
python preprocess_ocmr.py \
  --raw_dir /media/ndag/newVolume/ocmr_dataset/ocmr_cine/OCMR_data \
  --target_dir /media/ndag/newVolume/ocmr_dataset/ocmr_cine/OCMR_data_processed \
  --csv_query "smp=='fs'" --accelerations 8 12 16 20 --mask_types gro

```

### To start training:

```bash
cd flow_prior
python train.py --num_workers 4 --name prior-v1

```
