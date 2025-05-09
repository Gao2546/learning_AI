from utils.util import *
from utils.node import TransformerDecodeM

from utils.config import config

import torch
from flask import Flask, request, jsonify # Import Flask components

class Transformer_DecodeOnly:
    def __init__(self,  
                 tokenizer_path = "./models/BPEs/tokenizer-bpe-conversational-10k.json",
                 model_path = "./models/TransformerDecodeOnly/TransformerDecodeOnly_V01_256_768_12_12_3072_mn2_10K_MQcpk8.pth"
                ):
        self.tokenizer_path = tokenizer_path
        self.model_path = model_path
        self.llm = TransformerDecodeM(
                                        src_vocab_size=config["src_vocab_size"],
                                        tgt_vocab_size=config["tgt_vocab_size"],
                                        d_model=config["d_model"],
                                        num_heads=config["num_heads"],
                                        num_layers=config["num_layers"],
                                        d_ff=config["d_ff"],
                                        max_seq_length=config["max_seq_length"],
                                        dropout=config["dropout"],
                                        device=config["device"]
                                    )
        self.llm.to(config["device"])
        self.llm.eval()

        self.tokenizer = BPEsQA(config["tgt_vocab_size"])
        self.tokenizer.load(self.tokenizer_path)

        model_state_dict = torch.load(self.model_path)
        self.load_change_model(model_state_dict)

    def load_change_model(self, checkpoint):
        # Get the model's current state dict
        model_state_dict = self.llm.state_dict()

        # Filter out parameters that do not match in size
        filtered_state_dict = {
            k: v for k, v in checkpoint['model_state_dict'].items()
            if k in model_state_dict and v.size() == model_state_dict[k].size()
        }
        print("================No Match===============")
        [print(k) for k, v in checkpoint['model_state_dict'].items() if k in model_state_dict and v.size() != model_state_dict[k].size()]
        print("================No Match===============")

        # Update the model state dict with the filtered one
        model_state_dict.update(filtered_state_dict)

        # Load the updated state dict into the model
        self.llm.load_state_dict(model_state_dict)

    def invoke(self, question):
        answer_output = [1] + [5] + self.tokenizer.tokenizer.encode(question).ids + [6]
        start_seq = len(answer_output) - 1
        # print(start_seq)
        answer_output = torch.tensor(answer_output, device=config["device"])
        answer_output = torch.nn.functional.pad(answer_output,(0,config["max_seq_length"] - len(answer_output)),"constant",0).unsqueeze(0)
        # answer_output = torch.zeros((1,self.max_seq_length),device=self.device,dtype=torch.int32)
        # answer_output[0,0] = 1
        answer_input = answer_output.clone()
        seq_idx = 0
        with torch.no_grad():
            for seq_idx in range(start_seq,config["max_seq_length"] - 1):
                if answer_output[0].clone().cpu().tolist()[seq_idx] != 3:
                    answer_input[0,seq_idx] = answer_output.clone()[0,seq_idx]
                    answer_output = self.llm(answer_input)
                    answer_output = torch.argmax(torch.nn.functional.softmax(answer_output,dim=2),dim=2)
                else:
                    break
        answer_input[0,seq_idx] = answer_output.clone()[0,seq_idx]

        answer_question = self.tokenizer.decode_clean(answer_input[0,start_seq+1:seq_idx].cpu().tolist())

        return answer_question
    

    def serve(self, host='0.0.0.0', port=5000, debug=False):
        """
        Starts a Flask web server to provide an API endpoint for the model.
        """
        app = Flask(__name__)

        # It's good practice to disable Flask's default werkzeug logger
        # if you are using your own logging, or to avoid double logging with gunicorn.
        # For simplicity here, we'll leave it.

        @app.route('/health', methods=['GET'])
        def health_check():
            # Basic health check endpoint
            return jsonify({"status": "ok", "message": "Model server is running"}), 200

        @app.route('/predict', methods=['POST'])
        def predict():
            if not request.is_json:
                return jsonify({"error": "Request must be JSON"}), 400

            data = request.get_json()
            if 'question' not in data:
                return jsonify({"error": "Missing 'question' field in JSON payload"}), 400

            question = data['question']
            if not isinstance(question, str):
                return jsonify({"error": "'question' must be a string"}), 400
            
            if not question.strip():
                return jsonify({"error": "'question' cannot be empty or just whitespace"}), 400

            try:
                print(f"Received question: {question}")
                answer = self.invoke(question)
                print(f"Generated answer: {answer}")
                return jsonify({"answer": answer})
            except Exception as e:
                # Log the full error for debugging on the server side
                app.logger.error(f"Error during model invocation: {e}", exc_info=True)
                # Return a generic error to the client
                return jsonify({"error": "An internal error occurred during prediction."}), 500

        print(f"Starting Flask server on {host}:{port}...")
        # When deploying with Gunicorn/uWSGI, you wouldn't call app.run() directly.
        # The WSGI server would import 'app' from your file.
        # For development, app.run() is fine.
        # Setting threaded=True for slightly better handling of concurrent requests in dev.
        # For production, use a proper WSGI server like Gunicorn.
        app.run(host=host, port=port, debug=debug, threaded=True) 