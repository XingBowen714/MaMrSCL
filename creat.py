command_list = []

lrs = [5e-6, 1e-5, 2e-5, 3e-5]
l2s = [0.0, 0.01, 0.001, 1e-4]
other_lr_rate = [1,3,5]

dropouts = [0.1, 0.3, 0.5]
max_lens = [200]
schedual_lr_rates = [ 0.1,  0.3, 0.5, 0.7, 1.0]
lr_scheduals = ['cosine_decay', 'linear_decay']

cl_alphas = [1e-2, 1e-3, 1e-4]
neg_nums = [8, 10, 16,20, 30, 32, 40, 50, 60, 64, 80, 100]
temps = [0.05, 0.1]

mrd_alphas = [0.1, 1.0]


#for lr in lrs:
 #   for l2 in l2s:
  #      for hd in hidden_dims:
   #         for mcs in margin_cos_sent:
    #            for mca in margin_cos_act:
#constant lr
#command = 'CUDA_VISIBLE_DEVICES=0 python -u run.py --dataset WebNLG --batch_size 6 --num_train_epochs 200 --seed 2022' + ' --max_len ' + str(100)
#command = command + ' --bert_learning_rate ' + str(1e-5)+ ' --other_learning_rate ' + str(5e-5) + ' --lr_schedual constant --schedual_lr_rate -1 '

for n_neg in neg_nums:    
    #for cl_alpha in cl_alphas:
        #for t in temps:
        command = 'CUDA_VISIBLE_DEVICES=0 python -u run.py --dataset NYT --batch_size 64 --num_train_epochs 100 --seed 2022' + ' --weight_decay ' + str(0.0) + ' --dropout ' + str(0.1)
        command = command  + ' --cl_alpha ' + str(0.0001) + ' --num_neg_samples ' + str(n_neg) + ' --temperature ' + str(0.05) + ' --lr_schedual linear_decay' + \
            ' --schedual_lr_rate ' + str(0.5) + ' --mrd_alpha ' + str(0.1)
        command_list.append(command)
        #linear decay
    



num = 4
fns = ['run_nyt_{0}.sh'.format(i+1) for i in range(num)]
fs = [open(fn, 'w') for fn in fns]
for i in range(len(command_list)):
    b = i%num
    fs[b].write(command_list[i] + '\n')


for i in range(num):
    fs[b].close()
