"""
Inference and Generation Functions
Includes greedy and beam search decoding
"""

import torch
import math

def greedy_decode(model, src, src_mask, max_len, start_token, end_token, device):
    """Greedy decoding"""
    model.eval()
    
    with torch.no_grad():
        enc_output = model.encode(src, src_mask)
        tgt = torch.tensor([[start_token]], device=device)
        
        for _ in range(max_len):
            tgt_mask = (torch.triu(torch.ones((1, tgt.size(1), tgt.size(1)), device=device)) == 1).transpose(1, 2)
            tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
            
            dec_output = model.decode(tgt, enc_output, src_mask, tgt_mask.bool())
            output = model.fc_out(dec_output)
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            
            tgt = torch.cat([tgt, next_token], dim=1)
            
            if next_token.item() == end_token:
                break
        
        return tgt.squeeze(0).tolist()

def beam_search_decode(model, src, src_mask, max_len, start_token, end_token, device, beam_width=3):
    """Beam search decoding"""
    model.eval()
    
    with torch.no_grad():
        enc_output = model.encode(src, src_mask)
        beams = [([start_token], 0.0)]
        
        for step in range(max_len):
            new_beams = []
            
            for seq, score in beams:
                if seq[-1] == end_token:
                    new_beams.append((seq, score))
                    continue
                
                tgt = torch.tensor([seq], device=device)
                tgt_len = len(seq)
                tgt_mask = (torch.triu(torch.ones((1, tgt_len, tgt_len), device=device)) == 1).transpose(1, 2)
                tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
                
                dec_output = model.decode(tgt, enc_output, src_mask, tgt_mask.bool())
                output = model.fc_out(dec_output)
                probs = torch.softmax(output[:, -1, :], dim=-1)
                top_probs, top_indices = torch.topk(probs, beam_width)
                
                for prob, idx in zip(top_probs[0], top_indices[0]):
                    new_seq = seq + [idx.item()]
                    new_score = score - torch.log(prob).item()
                    new_beams.append((new_seq, new_score))
            
            beams = sorted(new_beams, key=lambda x: x[1])[:beam_width]
            
            if all(seq[-1] == end_token for seq, _ in beams):
                break
        
        return beams[0][0]

def generate_response(model, input_text, preprocessor, device, max_len=50, use_beam_search=True):
    """Generate chatbot response"""
    model.eval()
    
    input_normalized = preprocessor.normalize_urdu(input_text)
    if not input_normalized:
        return "معذرت، میں سمجھ نہیں سکا"
    
    input_indices = preprocessor.encode_sentence(input_normalized, max_len)
    src = torch.tensor([input_indices], device=device)
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    
    if use_beam_search:
        output_indices = beam_search_decode(
            model, src, src_mask, max_len,
            preprocessor.word2idx['<SOS>'],
            preprocessor.word2idx['<EOS>'],
            device,
            beam_width=3
        )
    else:
        output_indices = greedy_decode(
            model, src, src_mask, max_len,
            preprocessor.word2idx['<SOS>'],
            preprocessor.word2idx['<EOS>'],
            device
        )
    
    response = preprocessor.decode_sentence(output_indices)
    
    if not response.strip():
        response = "معذرت، میں جواب نہیں دے سکتا"
    
    return response