clear; close all; clc; more off
vocabMap = javaObject('java.util.Hashtable');
vocabList = getVocabList();
i = 1;
while i<=length(vocabList)
  vocabMap.put(vocabList{i}, i);
  i += 1;
end


files = glob('/home/akai/projects/learning/machine-learning-ex6/ex6/easy_ham/*.*');

fprintf('number of ham files %d', numel(files));
X = [];
y = [];
% for i=1:numel(files)
test_X = [];
test_y = [];

val_X = [];
val_y = [];
for i=1:200
  fprintf('loading email %d\n', i);
  [~, name] = fileparts(files{i});
  file = fileread(files{i});
  features = process_email(file, vocabMap);
  if i<150
    X = [X; features'];
    y = [y; 0];
  elseif i<180
    test_X = [test_X; features'];
    test_y = [test_y; 0];
  else
    val_X = [val_X; features'];
    val_y = [val_y; 0];
  end
end

 fflush(stdout);


files = glob('/home/akai/projects/learning/machine-learning-ex6/ex6/spam/*.*');

fprintf('number of spam files %d', numel(files));
for i=1:200
  fprintf('loading spam email %d\n', i);
  [~, name] = fileparts(files{i});
  file = fileread(files{i});
  features = process_email(file, vocabMap);
  if i<150
    X = [X; features'];
    y = [y; 1];
  elseif i<180
    test_X = [test_X; features'];
    test_y = [test_y; 1];
  else
    val_X = [val_X; features'];
    val_y = [val_y; 1];
  end
end

C = 0.1;
model = svmTrain(X, y, C, @linearKernel);
p = svmPredict(model, test_X);

fprintf('Training Accuracy: %f\n', mean(double(p == test_y)) * 100);

p = svmPredict(model, val_X);
fprintf('Training Accuracy validation: %f\n', mean(double(p == val_y)) * 100);
