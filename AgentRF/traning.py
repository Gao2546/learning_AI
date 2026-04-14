from utils.models import PPOAgentWeighted, PolicyNetworkModel, DQNAgent, PPOAgent

import torch.multiprocessing as mp
from utils.models import PPOAgent # and your other imports

if __name__ == '__main__':
    # 1. Set the start method FIRST
    mp.set_start_method('spawn', force=True)

    # Agent = PolicyNetworkModel(input_dim=None, hidden_dim=64, output_dim=None, dropout=0.5)

    # Agent.traningM()
    
    # Agent = DQNAgent()
    
    # Agent.training_loop()
    
    # 2. Initialize your Agent
    agent = PPOAgent(config_path="config.json", config_name="default")  # Adjust config_name as needed
    # agent = PPOAgentWeighted(config_path="config.json", config_name="default")  # Adjust config_name as needed
    
    # 3. Start the training
    # agent.training(checkpoint_filename="best_model.pth")  # You can specify a different checkpoint name if desired

    # 4. Start the evaluation (if needed)
    agent.test(total_episodes=100, checkpoint_name="best_model.pth")