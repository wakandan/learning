clear; close all; clc

files = glob('/home/akai/projects/learning/machine-learning-ex6/ex6/easy_ham/*.*');
fprintf('%d', numel(files));
X = [];
y = [];
for i=1:numel(files)
  fprintf('loading email %d\n', i);
  [~, name] = fileparts(files{i});
  file = fileread(files{i});
  content = processEmail(file);
  features = emailFeatures(content);
  X = [X; features'];
  y = [y; 0];
end
size(X)
size(y)
