nohup python3 /home/haochen/TPDM_transformer/DT/decision-transformer/collect_data.py>>collect_data_lt.out --scenario=left_turn 2>&1 &
nohup python3 /home/haochen/TPDM_transformer/DT/decision-transformer/collect_data.py>>collect_data_cr.out --scenario=cross 2>&1 &
nohup python3 /home/haochen/TPDM_transformer/DT/decision-transformer/collect_data.py>>collect_data_re.out --scenario=re 2>&1 &
nohup python3 /home/haochen/TPDM_transformer/DT/decision-transformer/collect_data.py>>collect_data_rm.out --scenario=rm 2>&1 &
nohup python3 /home/haochen/TPDM_transformer/DT/decision-transformer/collect_data.py>>collect_data_r.out --scenario=r 2>&1 &