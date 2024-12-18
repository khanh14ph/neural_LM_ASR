import torch
import math
from transformers import (
    AutoTokenizer, 
    AutoModelForMaskedLM, 
    DataCollatorForLanguageModeling,
    Trainer, 
    TrainingArguments
)
from datasets import load_dataset

class MaskedLanguageModelTrainer:
    def __init__(
        self, 
        model_name='vinai/phobert-base-v2', 
        
        max_length=128,
        dataset_name="filepath",
    ):
        # (Previous __init__ method remains the same)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        
        # Load and preprocess dataset
        self.dataset = self._prepare_dataset(
            dataset_name, 
            max_length
        ) 
    def _prepare_dataset(self, dataset_name, max_length):
        """
        Prepare and tokenize dataset
        
        Returns:
            Tokenized dataset ready for training
        """
        # Load raw dataset
        dataset = load_dataset(
                "csv",
                data_files=dataset_name,
                split="train",
                sep="\t",
                )
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['segment_text'], 
                truncation=True, 
                max_length=max_length, 
                padding='max_length'
            )
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            tokenize_function, 
            batched=True, 
        )
        
        return tokenized_dataset
    
    def train(
        self, 
        output_dir='./mlm_results',
        learning_rate=2e-5,
        batch_size=16,
        epochs=3
    ):
        """
        Train masked language model
        
        Args:
            output_dir (str): Directory to save model
            learning_rate (float): Training learning rate
            batch_size (int): Training batch size
            epochs (int): Number of training epochs
        """
        # Data collator for masked language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=True, 
            mlm_probability=0.15
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            logging_steps=10,
            save_steps=1000,
            save_total_limit=3,
            prediction_loss_only=True,
            learning_rate=learning_rate
        )
        
        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            data_collator=data_collator
        )
        
        # Train model
        trainer.train()
        
        # Save final model
        trainer.save_model()
    
    def predict_masked_tokens(self, text, mask_token_index):
        """
        Predict masked tokens and get their scores
        
        Args:
            text (str): Input text with masked token
            mask_token_index (int): Index of masked token
        
        Returns:
            Top k predicted tokens with their probabilities
        """
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits
        
        # Get predictions for masked token
        masked_token_logits = predictions[0, mask_token_index]
        probabilities = torch.softmax(masked_token_logits, dim=0)
        
        # Get top k predictions
        top_k = 5
        top_values, top_indices = torch.topk(probabilities, top_k)
        
        # Convert to readable tokens and probabilities
        predicted_tokens = self.tokenizer.convert_ids_to_tokens(top_indices.tolist())
        prediction_probs = top_values.tolist()
        
        return list(zip(predicted_tokens, prediction_probs))
    
    def score_sentence(self, sentence, reduction='mean'):
        """
        Score a sentence using the masked language model
        
        Args:
            sentence (str): Sentence to score
            reduction (str): How to reduce token-level scores 
                             ('mean', 'sum', or 'none')
        
        Returns:
            float or list: Sentence-level or token-level scores
        """
        # Tokenize the sentence
        inputs = self.tokenizer(sentence, return_tensors='pt').to(self.device)
        input_ids = inputs['input_ids'][0]
        
        # Create a copy of input_ids to track original tokens
        labels = input_ids.clone()
        
        # Compute token-level scores
        token_scores = []
        
        # Iterate through each token (except special tokens like [CLS], [SEP])
        for i in range(1, len(input_ids) - 1):
            # Create a copy of input_ids
            masked_input_ids = input_ids.clone()
            
            # Mask the current token
            masked_input_ids[i] = self.tokenizer.mask_token_id
            
            # Prepare masked input
            masked_inputs = {
                'input_ids': masked_input_ids.unsqueeze(0),
                'attention_mask': inputs['attention_mask']
            }
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**masked_inputs)
                logits = outputs.logits
            
            # Compute log likelihood of the original token
            log_likelihood = torch.log_softmax(logits[0][i], dim=0)[input_ids[i]]
            token_scores.append(log_likelihood.item())
        
        # Reduce scores based on specified method
        if reduction == 'mean':
            return sum(token_scores) / len(token_scores)
        elif reduction == 'sum':
            return sum(token_scores)
        else:
            return token_scores
    
    def compute_perplexity(self, text):
        """
        Compute perplexity of a given text
        
        Args:
            text (str): Text to compute perplexity for
        
        Returns:
            float: Perplexity score
        """
        # Tokenize the text
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        input_ids = inputs['input_ids'][0]
        
        # Create a copy of input_ids to track original tokens
        labels = input_ids.clone()
        
        # Compute token-level log probabilities
        log_likelihoods = []
        
        # Iterate through each token (except special tokens)
        for i in range(1, len(input_ids) - 1):
            # Create a copy of input_ids
            masked_input_ids = input_ids.clone()
            
            # Mask the current token
            masked_input_ids[i] = self.tokenizer.mask_token_id
            
            # Prepare masked input
            masked_inputs = {
                'input_ids': masked_input_ids.unsqueeze(0),
                'attention_mask': inputs['attention_mask']
            }
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**masked_inputs)
                logits = outputs.logits
            
            # Compute log likelihood of the original token
            log_likelihood = torch.log_softmax(logits[0][i], dim=0)[input_ids[i]]
            log_likelihoods.append(log_likelihood.item())
        
        # Compute perplexity
        avg_log_likelihood = sum(log_likelihoods) / len(log_likelihoods)
        perplexity = math.exp(-avg_log_likelihood)
        
        return perplexity

# Example usage
def main():
    # Initialize MLM
    mlm_trainer = MaskedLanguageModelTrainer(
        model_name='bert-base-uncased',
        dataset_name='/home4/khanhnd/neural_LM_ASR/indomain_corpus.tsv',
        max_length=128
    )
    mlm_trainer.train(
        output_dir='./mlm_results',
        learning_rate=2e-5,
        batch_size=16,
        epochs=3
    )
    # Example sentences for scoring
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming various industries.",
        "Natural language processing is a fascinating field."
    ]
    
    # Demonstrate different scoring methods
    print("Sentence Scoring Examples:")
    for sentence in sentences:
        # Token-level scores
        token_scores = mlm_trainer.score_sentence(sentence, reduction='none')
        print(f"\nSentence: {sentence}")
        print("Token-level Scores:")
        tokens = mlm_trainer.tokenizer.tokenize(sentence)
        for token, score in zip(tokens[1:-1], token_scores):
            print(f"{token}: {score:.4f}")
        
        # Aggregated scores
        mean_score = mlm_trainer.score_sentence(sentence, reduction='mean')
        print(f"Mean Score: {mean_score:.4f}")
        
        # Perplexity
        perplexity = mlm_trainer.compute_perplexity(sentence)
        print(f"Perplexity: {perplexity:.4f}")

if __name__ == '__main__':
    main()
