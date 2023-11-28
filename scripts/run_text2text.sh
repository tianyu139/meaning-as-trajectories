model=gpt2

# Prompt-free Results
for ds in stsbtest sts12 sts13 sts14 sts15 sts16 sickr; do
python text2text_main.py ${model} ${ds} trajectory --n_traj 20 --max_traj_len 20
done

# Prompted Results
for ds in stsbtest sts12 sts13 sts14 sts15 sts16 sickr; do
python text2text_main.py ${model} ${ds} trajectory --n_traj 20 --max_traj_len 20 --pre_text_prompt 'This sentence: "' --post_text_prompt '" implies: '
done
