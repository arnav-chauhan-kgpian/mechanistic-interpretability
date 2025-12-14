from pathlib import Path
import torch
import sys
from config import get_config, latest_weights_file_path
from model import build_transformer
from tokenizers import Tokenizer
from dataset import causal_mask

def translate(sentence: str):
    # 1. Load Config and Device
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Tokenizers
    # We load the existing files (we do NOT retrain them)
    tokenizer_src = Tokenizer.from_file(config['tokenizer_file'].format(config['lang_src']))
    tokenizer_tgt = Tokenizer.from_file(config['tokenizer_file'].format(config['lang_tgt']))

    # 3. Build Model
    model = build_transformer(
        tokenizer_src.get_vocab_size(), 
        tokenizer_tgt.get_vocab_size(), 
        config['seq_len'], 
        config['seq_len'], 
        d_model=config['d_model']
    ).to(device)

    # 4. Load Pre-trained Weights
    model_filename = latest_weights_file_path(config)
    if not model_filename:
        print("No weights found! Train the model first.")
        return
    
    print(f"Loading weights from: {model_filename}")
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval() # Switch to evaluation mode

    # 5. Prepare Input Text
    # Encode the sentence and add SOS/EOS
    sos_token = tokenizer_tgt.token_to_id('[SOS]')
    eos_token = tokenizer_tgt.token_to_id('[EOS]')
    
    encoder_input_tokens = tokenizer_src.encode(sentence).ids
    encoder_input = torch.tensor(
        [sos_token] + encoder_input_tokens + [eos_token], 
        dtype=torch.int64
    ).to(device)
    
    # Add batch dimension (1, seq_len)
    encoder_input = encoder_input.unsqueeze(0) 
    
    # Create Encoder Mask (1, 1, 1, seq_len)
    # Since we are doing inference on 1 sentence, we don't strictly need padding mask if we handle lengths right,
    # but strictly speaking, we should mask padding if we had it. Here we have no padding.
    encoder_mask = (encoder_input != tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)

    # 6. Run Encoder
    with torch.no_grad():
        encoder_output = model.encode(encoder_input, encoder_mask)

        # 7. Auto-regressive Decoder (Greedy Decode)
        # Start with just [SOS]
        decoder_input = torch.empty(1, 1).fill_(sos_token).type_as(encoder_input).to(device)

        while True:
            if decoder_input.size(1) == config['seq_len']:
                break

            # Create Mask for Decoder
            decoder_mask = causal_mask(decoder_input.size(1)).type_as(encoder_mask).to(device)

            # Calculate output
            out = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)

            # Get next token probability
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)

            # Append next word to decoder input
            decoder_input = torch.cat(
                [decoder_input, torch.empty(1, 1).type_as(encoder_input).fill_(next_word.item()).to(device)], 
                dim=1
            )

            if next_word == eos_token:
                break

    # 8. Decode the result back to text
    # Squeeze batch dim, remove SOS
    output_ids = decoder_input.squeeze(0).tolist()
    
    translated_text = tokenizer_tgt.decode(output_ids)
    print(f"\nEnglish: {sentence}")
    print(f"Hindi:   {translated_text}")

if __name__ == '__main__':
    # Usage: python translate.py "Hello world"
    if len(sys.argv) > 1:
        sentence = sys.argv[1]
        translate(sentence)
    else:
        print("Please provide a sentence. Example: python translate.py 'Hello world'")