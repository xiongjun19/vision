set -x 
log_dir=logs

mkdir -p ${log_dir}

bs_arr=(64)
card_num_arr=(1 2 4 6)
for(( i=0;i<${#bs_arr[@]};i++)) do
   for(( j=0;j<${#card_num_arr[@]};j++)) do
       bs=${bs_arr[i]};
       card_num=${card_num_arr[j]};
       cpu_log="${log_dir}/${card_num}_logs_local.txt"
       gpu_log="${log_dir}/res50_card${card_num}_bs${bs}.qdrep" 
       python -m torch.distributed.launch --nproc_per_node=${card_num} \
	       resnet_train.py -i /workspace/data/train_t3 --batch_size ${bs} \
	       --epochs 2 --num_work 4 > ${cpu_log};

       nsys profile  -c cudaProfilerApi -f true --stats true  -o ${gpu_log} \
	       python -m torch.distributed.launch --nproc_per_node=${card_num} \
	       resnet_train_prof.py -i /workspace/data/train_t3 --batch_size ${bs} \
	       --epochs 2 --num_work 4
       
   done
done
 
