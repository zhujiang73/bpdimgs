@echo off

set  time01=%time%
set  TOOLS=c:\mingw\bin

%TOOLS%\caffe.exe  train  --solver=.\data\solver.prototxt

set  time02=%time%

echo time 01:  %time01%
echo time 02:  %time02%
