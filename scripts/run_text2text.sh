model=gpt2

# Prompt-free Results
for ds in stsbtest sts12 sts13 sts14 sts15 sts16 sickr; do
python text2text_main.py ${model} ${ds} trajectory 
done

# Prompted Results
for ds in stsbtest sts12 sts13 sts14 sts15 sts16 sickr; do
python text2text_main.py ${model} ${ds} trajectory --pre_text_prompt 'This sentence: "' --post_text_prompt '" implies: '
done
