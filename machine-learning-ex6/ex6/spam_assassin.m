clear; close all; clc

files = glob('/home/akai/projects/learning/machine-learning-ex6/ex6/easy_ham/*.*');
printf("%d", numel(files));
for i=1:numel(files)
  [~, name] = fileparts(files{i});
  file = fileread(files{i});
  lines = strsplit(file, '\n');
  disp(lines);
  for line=lines
    printf("%s", line);
    //read until gets the header From:
    if(regexp(line, '^From: .*<.*@.*>$'))
      //start stripping other headers
      
    end
  end
  break;
end
