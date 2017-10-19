set  TOOLS=c:\mingw\bin
set  IMGSDIR=..\images
set  IMGSIZE=137

%TOOLS%\convert_imageset.exe  --gray  --resize_height=%IMGSIZE%  --resize_width=%IMGSIZE%   %IMGSDIR%\bpcv_train\  .\data\train.txt  .\train_lmdb

%TOOLS%\convert_imageset.exe  --gray  --resize_height=%IMGSIZE%  --resize_width=%IMGSIZE%   %IMGSDIR%\bpcv_test\   .\data\val.txt    .\test_lmdb


