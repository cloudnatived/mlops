import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size=50257, hidden_size=768, num_layers=12, 
                 num_heads=12, max_seq_length=1024):
        super(SimpleTransformer, self).__init__()
        
        # 使用Hugging Face的GPT2配置
        config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=hidden_size,
            n_layer=num_layers,
            n_head=num_heads,
            n_positions=max_seq_length,
            n_ctx=max_seq_length,
        )
        
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # 权重绑定
        self.lm_head.weight = self.transformer.wte.weight
        
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        # 前向传播
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # 计算交叉熵损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return {
            'logits': logits,
            'loss': loss,
            'hidden_states': hidden_states
        }
    
    def generate(self, input_ids, max_length=100, temperature=1.0):
        """简单的生成函数"""
        self.eval()
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                outputs = self.forward(input_ids)
                next_token_logits = outputs['logits'][:, -1, :] / temperature
                
                # 采样
                next_token = torch.multinomial(
                    F.softmax(next_token_logits, dim=-1), num_samples=1
                )
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # 如果生成了结束符，停止生成
                if next_token.item() == 50256:  # GPT2的结束符
                    break
        
        return input_ids

def create_model(args):
    """创建模型实例"""
    model = SimpleTransformer(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_length=args.max_seq_length
    )
    return model
