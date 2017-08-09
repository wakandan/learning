function features = process_email(email_contents, vocabMap)
word_indices = [];
hdrstart = strfind(email_contents, ([char(10) char(10)]));
email_contents = email_contents(hdrstart(1):end);
email_contents = lower(email_contents);
email_contents = regexprep(email_contents, '<[^<>]+>', ' ');
email_contents = regexprep(email_contents, '[0-9]+', 'number');
email_contents = regexprep(email_contents, ...
                           '(http|https)://[^\s]*', 'httpaddr');
email_contents = regexprep(email_contents, '[^\s]+@[^\s]+', 'emailaddr');
email_contents = regexprep(email_contents, '[$]+', 'dollar');
l = 0;
features= zeros(vocabMap.size(), 1);
while ~isempty(email_contents)

    % Tokenize and also get rid of any punctuation
    [str, email_contents] = ...
       strtok(email_contents, ...
              [' @$/#.-:&*+=[]?!(){},''">_<;%' char(10) char(13)]);

    % Remove any non alphanumeric characters
    str = regexprep(str, '[^a-zA-Z0-9]', '');

    % Stem the word
    % (the porterStemmer sometimes has issues, so we use a try catch block)
    try str = porterStemmer(strtrim(str));
    catch str = ''; continue;
    end;

    % Skip the word if it is too short
    if length(str) < 1
       continue;
    end
  features(vocabMap.get(str)) = 1;
end
end
