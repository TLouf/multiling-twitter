import re
# Since cld3 can't be installed on some systems, we make it optional.
try:
    import cld3
    DEFAULT_CLD = 'cld3'
except ModuleNotFoundError:
    DEFAULT_CLD = 'pycld2'
import pycld2
import src.utils.my_exceptions as my_exceptions

def lang_detect(tweets_df, text_col='text', min_nr_words=4, min_nr_cjk=4,
                cld=DEFAULT_CLD, acc_th=0.9, langs_agg_dict=None):
    '''
    From a DataFrame of tweets, detects the language of the text contained in
    'text_col' and output the result in new columns. Before calling a langauge
    detector, the text is pre-processed to rid it of hashtags, usernames in
    mentions and urls. Also, after these are removed, if the remaining text
    is strictly less than 'min_nr_words' words long, we don't bother calling the
    language detector, which will be unreliable anyway.
    '''
    tweets_lang_df = tweets_df.copy()
    # Match anything starting with @ or # followed by one or more word
    # characters, and which is between word boundaries or at the start or end of
    # the string.
    hash_at_pattern = r'(?:^|\B)((@|#)\w+)(?:$|\b)'
    # Match anything containing /t.co/ surrounded by non-whitespace characters
    # and which is between whitespaces or at the start or end of the string. May
    # be better with all http options at the start, here's it's pretty loose.
    url_pattern = r'(?:^|\s)(\S+\/t.co\/\S+)(?:$|\b)'
    regex_filter = re.compile('({})|({})'.format(hash_at_pattern, url_pattern))
    tweets_lang_df['filtered_text'] = tweets_lang_df[text_col].str.replace(
        regex_filter, '')
    foursquare_mask = tweets_lang_df['source'].str.contains('foursquare')
    im_at_mask = tweets_lang_df['text'].str.startswith("I'm at ")
    # These tweets are generated by Foursquare based on people's location, so
    # they don't tell us anything about the user's language.
    tweets_lang_df.loc[foursquare_mask & im_at_mask, 'filtered_text'] = ''
    # Tweets generated by Foursquare can also end with '(@ <location>)', so we
    # get rid of this part which is useless for language detection, as proper
    # nouns just confuse the detector.
    tweets_lang_df.loc[foursquare_mask, 'filtered_text'] = (
        tweets_lang_df.loc[foursquare_mask, 'filtered_text']
                      .str.replace(r'\(@ .*$', '')
    )
    # Tweets generated by Instagram can end with ' @ <location>', so we
    # get rid of this part which is useless for language detection, as proper
    # nouns just confuse the detector.
    insta_mask = tweets_lang_df['source'].str.contains('instagram')
    tweets_lang_df.loc[insta_mask, 'filtered_text'] = (
        tweets_lang_df.loc[insta_mask, 'filtered_text']
                      .str.replace(r' @ .*$', '')
    )
    # Tweets generated by Path can contain information about the location
    # (after a 'at') and/or the people the user was with (after a 'with'). This
    # metadata is either in parentheses after the core of the tweet, or, if
    # there's is no core text, right at the start without parentheses. Thus we
    # extract only the core text from what's before a '(at' or '(with'.
    path_mask = tweets_lang_df['source'].str.contains('path')
    tweets_lang_df.loc[path_mask, 'filtered_text'] = (
        tweets_lang_df.loc[path_mask, 'filtered_text']
                      .str.extract(r'(.*)(?:\(with|\(at)', expand=False)
    )
    # Extract returns NaN if there's no match, so we need to convert these
    # to the empty string to avoid errors.
    tweets_lang_df.loc[tweets_lang_df['filtered_text'].isnull(),
                       'filtered_text'] = ''
    # A word is defined by the shortest string of letters between two word
    # boundaries, but a letter here is not simply a-z, because we also want to
    # account for non latin alphabets. A letter is then what is neither a digit
    # (\d), nor an underscore, nor a non-word character (\W: punctuation,
    # special characters, emojis...).
    word_any_lang_pattern = re.compile(r'\b[^\W\d_]+?\b')
    nr_words = tweets_lang_df['filtered_text'].str.count(word_any_lang_pattern)
    # We also count the number of Chinese-Japanese-Korean (CJK) characters,
    # because they can be a full word or syllable, and a whole sentence can be
    # written without any space, so the word count is irrelevant.
    cjk_chars_pattern = re.compile(r'[\u4E00-\u9FFF]')
    nr_cjk_chars = tweets_lang_df['filtered_text'].str.count(cjk_chars_pattern)
    long_enough = (nr_words >= min_nr_words) | (nr_cjk_chars >= min_nr_cjk)
    (tweets_lang_df.loc[long_enough, 'cld_lang'],
     tweets_lang_df.loc[long_enough, 'proba']) = zip(
        *tweets_lang_df.loc[long_enough, 'filtered_text'].apply(
            lambda t: make_predict(t, cld=cld, langs_agg_dict=langs_agg_dict)))
    acc_mask = tweets_lang_df['proba'] < acc_th
    un_mask = tweets_lang_df['cld_lang'] == 'un'
    tweets_lang_df.loc[acc_mask | un_mask, 'cld_lang'] = None
    return tweets_lang_df



def make_predict(text, cld=DEFAULT_CLD, langs_agg_dict=None):
    '''
    From a string of text, gets the language detection from one of the CLD
    versions, and unpacks the results in a dictionary.
    '''
    if langs_agg_dict is None:
        langs_agg_dict = {}
    if  cld == 'cld3':
        raw_predict = cld3.get_language(text)
        lang = raw_predict.language
        proba = raw_predict.probability
    elif cld == 'pycld2':
        # Sometimes there remain utf code like \x92, which makes pycld2 return
        # an error, so we skip these tweets (there are very few of them)
        try:
            raw_predict = pycld2.detect(text)
            lang = raw_predict[2][0][1]
            proba = raw_predict[2][0][2] / 100
        except pycld2.error:
            lang = None
            proba = 0
    else:
        raise my_exceptions.InputError(cld, ['cld3', 'pycld2'])
    # The following is for if the lang is to be aggregated with a similar one.
    # For instance, we might like to aggregate to Chinese ('zh') its traditional
    # form ('zh-Hant').
    agg_to_lang = langs_agg_dict.get(lang)
    if agg_to_lang is not None:
        lang = agg_to_lang
    return lang, proba
