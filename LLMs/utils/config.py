# config.py

config = {
    "src_vocab_size": 10240,
    "tgt_vocab_size": 10240,
    "d_model": 128*3,
    "num_heads": 6*1,
    "num_layers": 6*1,
    "d_ff": 128*3*4,
    "max_seq_length": 256,
    "dropout": 0.1,
    "device": "cuda"  # or "cpu"
}
