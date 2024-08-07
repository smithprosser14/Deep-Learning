import pandas as pd
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    pipeline,
    PreTrainedModel,
    PreTrainedTokenizerBase
)


class Translator:
    """
    Accepts a filepath to a movie reviews dataframe of a given language.

    Methods:
        1. language_identifier: Identifies the language of the movie reviews.
        2. _obtain_data: Obtains and pre-processes the data from specified csv file.
        3. _model_set_up: Sets up model/tokenizer for identified language.
        4. _translate_text: Translates a piece of text into english.
        5. translate: Translates all the "Synopsis" and "Review" into english.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.original_df = None

    @staticmethod
    def language_identifier(text: str):
        lang_id_model_name = "papluca/xlm-roberta-base-language-detection"
        lang_id_pipe = pipeline(task="text-classification", model=lang_id_model_name)

        lang = lang_id_pipe(text)

        return lang

    def _obtain_data(self) -> (pd.DataFrame, str):
        # Note: All the dataframes would have different column names. For testing purposes
        # you should have the following column names/headers -> [Title, Year, Synopsis, Review]
        col_names = ['Title', 'Year', 'Synopsis', 'Review']

        # ingest the data from relevant filepath
        raw_df = pd.read_csv(self.filepath).reset_index(drop=True)
        # rename columns for formatting purposes
        raw_df.columns = col_names
        self.original_df = raw_df
        df = self.original_df.copy()

        # The language of the reviews are detected identified using a language identification model
        lang = self.language_identifier(df["Review"][0])[0]["label"]

        return df, lang

    @staticmethod
    def _model_set_up(lang: str) -> (PreTrainedModel, PreTrainedTokenizerBase):
        """
        The model and tokenizer are chosen based on the detected language of the reviews.
        """
        model_name = f"Helsinki-NLP/opus-mt-{lang}-en"
        model = MarianMTModel.from_pretrained(model_name)
        tokenizer = MarianTokenizer.from_pretrained(model_name)

        return model, tokenizer

    @staticmethod
    def _translate_text(text: str, model, tokenizer) -> str:
        """
        Function to translate a piece of text using a model and tokenizer.

        Arg:
            text: string of text in a language besides english.
            model: model object.
            tokenizer = tokenizer object.

        Returns:
            String translated into english.
        """
        # encode the text using the tokenizer - returns embedding + attention mask
        inputs = tokenizer(text, return_tensors="pt")

        # generate the translation using the model - returns output as embedding
        outputs = model.generate(**inputs)

        # decode the generated output and return the translated text - converts embedding back to text
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return decoded

    def translate(self) -> pd.DataFrame:
        """
        Obtains the data from the given filepath, runs-preprocessing steps, decides model and tokenizer, and
        carries out the translation of the "Synopsis" and "Review".

        Returns:
            Dataframe where the "Synopsis" and "Review" columns now contain the english translations.
        """
        processed_df, lang = self._obtain_data()

        print(f"Original Language: {lang.upper()}")

        model, tokenizer = self._model_set_up(lang)

        processed_df["Review"] = processed_df["Review"].apply(self._translate_text, args=(model, tokenizer))
        processed_df["Synopsis"] = processed_df["Synopsis"].apply(self._translate_text, args=(model, tokenizer))

        translated_df = processed_df.copy()

        return translated_df
