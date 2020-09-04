epoch_count = 8
python_cmd = `if [ "$$(uname -s)" = "Darwin" ]; then \
	echo "python3"; \
else \
	echo "python"; \
fi;` \

gen-dataset:
	$(python_cmd) datasets/gen_captcha.py -d --npi=4 -n $(epoch_count)

train:
	$(python_cmd) cnn_n_char.py  --data_dir images/char-4-epoch-$(epoch_count)/
