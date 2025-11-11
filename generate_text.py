from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# generates text based on user prompt and sentiment
class generate_text:
    def __init__(self, device: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")

        # gpt 2 does not have a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModelForCausalLM.from_pretrained('gpt2-medium').to(self.device)

    def generate(self, prompt, sentiment: str = 'NEUTRAL', max_length: int = 150,
                 temperature: float = 0.9, num_return_sequences: int = 1):
        
        # sentiments to be given as prompts to the model
        sentiments = {
            'POSITIVE':'Write a happy, motivational, calm and feel good paragrah about:',
            'NEGATIVE':'Write a serious and critical paragraph about:',
            'NEUTRAL': 'Write a normal, neutral and balanced paragraph about:',
        }.get(sentiment, 'Write a normal, neutral and balanced paragraph about:')

        # combine sentiments with the 
        full_prompt = f"{sentiments} {prompt}.\n\n"
        input_ids = self.tokenizer(full_prompt, return_tensors="pt").input_ids.to(self.device)

        # generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_length=min(len(input_ids[0]) + max_length, 1024),
                temperature=temperature,
                do_sample=True,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        results = []
        for i in range(num_return_sequences):
            gen = outputs[i]
            # decode and remove the prompt part
            text = self.tokenizer.decode(gen, skip_special_tokens=True)
            # Strip the prefix prompt from the output if it's repeated
            if text.startswith(full_prompt):
                text = text[len(full_prompt):].strip()
            results.append(text)

        return results if num_return_sequences > 1 else results[0]
    


