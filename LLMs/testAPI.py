from utils import llm

model = llm.Transformer_DecodeOnly(
                tokenizer_path = "./models/BPEs/tokenizer-bpe-conversational-10k.json",
                model_path = "./models/TransformerDecodeOnly/TransformerDecodeOnly_V01_256_768_12_12_3072_10K_MQcpk9NT.pth"
)

if __name__ == "__main__":
    model.serve()