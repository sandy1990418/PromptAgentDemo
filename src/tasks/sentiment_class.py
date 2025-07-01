import re
from datasets import load_dataset
from .base_task import BaseTask

class CustomTask(BaseTask):
    def __init__(self, 
                 train_size, 
                 eval_size,
                 test_size=None,  
                 task_name = "sentiment_class",
                 task_description = "sentiment classification",
                 data_dir='',  
                 seed=None, 
                 
                 post_instruction=True, 
                 **kwargs):
        self.options = {}
        super().__init__(
                        task_name = task_name,  
                        task_description = task_description, 
                        data_dir=data_dir,
                        seed = seed,
                        train_size = train_size,
                        eval_size=eval_size,
                        test_size = test_size,
                        post_instruction = post_instruction,
                        )

        self.answer_format_prompt = "\nA:"
    
    def load_task_dataset(self, data_dir):
        '''
            <task specific>
        '''
        json_data = self._load_json_file(data_dir)
        self.task_description = """Classify sentiment of the text as 'positive' or 'negative'."""
        return json_data
    
    def transform_format(self, data):
        original_examples = data['examples']
        examples = []
        # Extracting input and target scores
        for example in original_examples:
            question = example['text']            
            # Generating options and answer            
            answer = example['sentiment']

            question_str = "Classify sentiment of the text as 'positive' or 'negative'.'.\n"+question
            
            # Formatting the output
            formatted_example = {
                'question': question_str,
                'answer': answer
            }
            examples.append(formatted_example)
        
        return examples
    
    def clean_response(self, response):
        clean_pattern = r"\b(positive|negative)\b"
        match = re.findall(clean_pattern, response.lower())
        if len(match) != 0:
            return match[-1]
    
        return "N/A: format error."
    