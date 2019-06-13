from __future__ import unicode_literals, print_function
import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding

LABEL = 'position'

TRAIN_DATA = [
    ("Hello, could you tell me who is head of IT.", {
        'entities': [(-11, -1, LABEL)]
    }),

    ("Man, what\'s up. I need some information from head of advisor, do you know who she/he is?", {
        'entities': [(45, 60, LABEL)]
    }),

    ("In yesterday, a new colleague join in our group. He is IT engineer.", {
        'entities': [(-12, -1, LABEL)]
    }),

    ("Our clients need some advise from HR, could you give me a list in Stuttgart office.", {
        'entities': [(34, 36, LABEL)]
    }),

    ("At the Stuttgart office, our tax lawyers have a meeting for new business.", {
        'entities': [(29, 41, LABEL)]
    }),

    ("In general, we need to hire an assistant for each lawyer.", {
        'entities': [(31, 41, LABEL)]
    }),

    ("You are good employee, good IT engineer.", {'entities': [(-12, -1, LABEL)]}),

    ("There is three tax lawyer join our group in March 2018.", {
        'entities': [(15, 25, LABEL)]
    }),

    ("Our company currently recruits accountant. The basic requirement is three years of work experience.", {
        'entities': [(31, 41, LABEL)]
    }),

    ("Do you know how many HR our company has?", {
        'entities': [(21, 23, LABEL)]
    }),

    ("Our auditors can provide you with trusted service.", {
        'entities': [(4, 12, LABEL)]
    }),

    ("I need a tax lawyer to advise me about my debt problem.", {
        'entities': [(9, 19, LABEL)]
    }),

    ("I have never seen the CEO of the company.", {
        'entities': [(22, 25, LABEL)]
    }),

    ("Do you know where the CTO office?", {
        'entities': [(-11, -8, LABEL)]
    }),

    ("Our company\'s COO has been working for the company for 10 years.", {
        'entities': [(14, 17, LABEL)]
    }),

    ("Is there a problem with your computer? Unfortunately, our IT engineers are sick leave in this week.", {
        'entities': [(58, 70, LABEL)]
    }),

    ("Do you know how to win a head of IT position in one company?", {
        'entities': [(33, 44, LABEL)]
    }),

    ("The company\'s tax lawyer are discussing this new business in the meeting room.", {
        'entities': [(14, 25, LABEL)]
    }),

    ("Are the company\'s assistants at work today?", {
        'entities': [(18, 27, LABEL)]
    }),

    ("This is the assistant office and the largest office.", {
        'entities': [(12, 21, LABEL)]
    }),

    ("How can I become a HR?", {
        'entities': [(-3, -1, LABEL)]
    }),

    ("I made an appointment with a auditor next Friday.", {
        'entities': [(29, 37, LABEL)]
    }),

    ("The company has plans to train a new tax lawyer.", {
        'entities': [(-11, -1, LABEL)]
    }),
]


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int))
def main(model=None, new_model_name='position',
         output_dir='Date_NER_folder', n_iter=25):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe('ner')

    ner.add_label(LABEL)
    if model is None:
        optimizer = nlp.begin_training()
    else:
        # Note that 'begin_training' initializes the models, so it'll zero out
        # existing entity types.
        optimizer = nlp.entity.create_optimizer()

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.30,
                           losses=losses)
            print('Losses', losses)

    # test the trained model
    test_text = 'Hi, tell me tax lawyer information.'
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta['name'] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)


if __name__ == '__main__':
    plac.call(main)
