set  IMGSDIR=..\images

python3   imgs_fn_list.py    %IMGSDIR%\bpcv_train    .\data\train.txt
python3   imgs_fn_list.py    %IMGSDIR%\bpcv_test     .\data\val.txt

rem  pause
