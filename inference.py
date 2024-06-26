import torch
import pickle
from config import *
from model.rnn import RNN
from utils.tokenizer import CharacterTokenizer

def generate_text(model: RNN, tokenizer: CharacterTokenizer, seed_text: str, num_chars: int, device: torch.device) -> str:
    model.eval()
    generated_text = seed_text
    hidden = None

    with torch.no_grad():
        for _ in range(num_chars):
            input_seq = torch.tensor(tokenizer.encode(generated_text[-SEQUENCE_LENGTH:])).unsqueeze(0).to(device)
            output, hidden = model(input_seq, hidden)
            probabilities = output[0, -1].softmax(dim=-1)
            next_char_idx = torch.multinomial(probabilities, num_samples=1).item()
            generated_text += tokenizer.decode([next_char_idx])

    return generated_text

def inference_shakespeare_model(input_text: str, num_chars: int = 200) -> str:
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    
    checkpoint = torch.load(MODEL_PATH)
    model = RNN(checkpoint['vocab_size'], EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return generate_text(model, tokenizer, input_text, num_chars, DEVICE)

def main_inference():
    prompts = [
        "O Romeo, Romeo",
        "To be, or not to be",
        "All the world's a stage",
        "Friends, Romans, countrymen",
    ]

    print("\n--- Generating Shakespeare-like text using the trained model ---")
    for prompt in prompts:
        generated_text = inference_shakespeare_model(prompt)
        print(f"\nPrompt: {prompt}")
        print(f"Generated text:\n{generated_text}\n")
        print("-" * 50)
    
    while True:
        user_input = input("\nEnter your own prompt (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        generated_text = inference_shakespeare_model(user_input)
        print(f"\nGenerated text:\n{generated_text}\n")
        print("-" * 50)

if __name__ == "__main__":
    main_inference()