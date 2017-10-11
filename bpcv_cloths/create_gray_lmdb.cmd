set  TOOLS=c:\mingw\bin

%TOOLS%\convert_imageset.exe  --gray  --resize_height=120  --resize_width=120  ..\images\bpcv_train\  .\data\train.txt  .\train_lmdb

%TOOLS%\convert_imageset.exe  --gray  --resize_height=120  --resize_width=120  ..\images\bpcv_test\   .\data\val.txt    .\test_lmdb


