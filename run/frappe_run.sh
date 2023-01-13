python ../code/main.py \
    --dataset=frappe \
    --dim=32 \
    --out_dim=64 \
    --hidden_size=128 \
    --cl_w=0.1 \
    --dis_w=0.1 \
    --lr=0.001 \
    --merge=prod \
    --temp=0.5 \
    --n_neg_cl=40 \
    --n_neg_dis=5 \
    --infomin_step=5 \
    --vis_emb=0 \
    --summary='' \