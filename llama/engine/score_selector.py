from llama.engine.base_selector import BaseSelector


class ScoreSelector(BaseSelector):
    def __init__(self):
        super().__init__()
        self.higher_is_better = True
    
    def sort(self, examples, _):
        scored_examples, unscored_examples = [], []
        for e in examples:
            if 'score' in e:
                scored_examples.append(e)
            else:
                unscored_examples.append(e)
        sorted_examples = sorted(scored_examples, key=lambda elt: elt['score'], reverse=self.higher_is_better)
        examples = sorted_examples + unscored_examples
        return examples
