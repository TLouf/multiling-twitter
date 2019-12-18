# Since cld3 can't be installed on some systems, we make it optional.
try:
    import cld3
    default_cld = 'cld3'
except ModuleNotFoundError:
    default_cld = 'pycld2'
import pycld2
import pandas as pd
import re
import src.utils.my_exceptions as my_exceptions

def lang_detect(tweets_df, text_col='text', min_nr_words=4, cld=default_cld):
    '''
    From a DataFrame of tweets, detects the language of the text contained in
    'text_col' and output the result in new columns. Before calling a langauge
    detector, the text is pre-processed to rid it of hashtags, usernames in
    mentions and urls. Also, after these are removed, if the remaining text
    is strictly less than 'min_nr_words' words long, we don't bother calling the
    language detector, which will be unreliable anyway.
    '''
    # Match anything starting with @ or # followed by one or more word
    # characters, and which is between word boundaries or at the start or end of
    # the string.
    hash_at_pattern = r'(?:^|\B)((@|#)\w+)(?:$|\b)'
    # Match anything containing /t.co/ surrounded by non-whitespace characters
    # and which is between whitespaces or at the start or end of the string may
    # be better with all http options at the start, here's it's pretty loose.
    url_pattern = r'(?:^|\s)(\S+\/t.co\/\S+)(?:$|\b)'
    # Sometimes there remain utf code like \x92, which makes pycld2 return an
    # error. re can't find it in the str object, so we have to get its repr.
    # Then to have a "normal" str again, we need to eval the result.
    if cld == 'pycld2':
        utfcode_pattern = r'\\x[a-f0-9]{2}'
        utfcode_regex = re.compile(utfcode_pattern)
        tweets_df[text_col] = tweets_df[text_col].apply(
            lambda x: eval(re.sub(utfcode_regex, '', repr(x))))
    regex_filter = re.compile('({})|({})'.format(hash_at_pattern, url_pattern))
    tweets_df['filtered_text'] = tweets_df[text_col].str.replace(regex_filter,'')
    # A word is defined by the shortest string of letters between two word
    # boundaries, but a letter here is not simply a-z, because we also want to
    # account for non latin alphabets. A letter is then what is neither a digit
    # (\d), nor an underscore, nor a non-word character (\W: punctuation,
    # special characters, emojis...).
    word_any_lang_pattern = re.compile(r'\b[^\W\d_]+?\b')
    nr_words = tweets_df['filtered_text'].str.count(word_any_lang_pattern)
    long_enough = nr_words >= min_nr_words
    new_tweets_df = tweets_df.loc[long_enough]
    # Then we create a new DataFrame by detecting the language of the filtered
    # text, and expanding the results of the prediction, which is a dictionary,
    # in new columns.
    filtered_text_array = new_tweets_df['filtered_text'].values
    preds = [make_predict(t, cld=cld) for t in filtered_text_array]
    predict_df = pd.DataFrame(preds, index=new_tweets_df.index)
    new_tweets_df = pd.concat([new_tweets_df, predict_df], axis=1)
    return new_tweets_df


def make_predict(text, cld=default_cld):
    '''
    From a string of text, gets the language detection from one of the CLD
    versions, and unpacks the results in a dictionary.
    '''
    if  cld == 'cld3':
        raw_predict = cld3.get_language(text)
        lang = raw_predict.language
        proba = raw_predict.probability
    elif cld == 'pycld2':
        raw_predict = pycld2.detect(text)
        lang = raw_predict[2][0][1]
        proba = raw_predict[2][0][2]
    else:
        raise my_exceptions.InputError(cld, ['cld3', 'pycld2'])
    predict = {'cld_lang': lang, 'proba': proba}
    return predict
