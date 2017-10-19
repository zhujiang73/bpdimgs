import sys
import os
import os.path

imgs_dir = sys.argv[1]
caffe_imgs_fn = sys.argv[2]

mydirs = []

for parent,dirnames,filenames in os.walk(imgs_dir): #{
	for dirname in dirnames: #{
		mydirs.append(dirname)
        #}
#}

mydirs.sort()
n = len(mydirs)
for i in range(0, n):
	str_buf = mydirs[i]
	print(str_buf)

list_dicts = {}
list_files = []
for i in range(0, len(mydirs)):
	list_files = []
	mydir = os.path.join(imgs_dir,mydirs[i])
	for parent,dirnames,filenames in os.walk(mydir):
		for filename in filenames:
			list_files.append(os.path.join(mydirs[i],filename))
	list_dicts[mydirs[i]] = list_files

num_lens = []
for mydir in mydirs:	
	file_lists = list_dicts[mydir]
	num_len = len(file_lists)
	num_lens.append(num_len)

file_imgs = open(caffe_imgs_fn, 'w')

num_files = min(num_lens)
num_dirs = len(mydirs)
for nf in range(0,num_files):
	for nd in range(0, num_dirs):
		list_files = list_dicts[mydirs[nd]]
		file_name = list_files[nf]
		str_buf = "%s  %d"%(file_name, nd)
		print(str_buf)
		str_buf += "\n"
		file_imgs.write(str_buf)	

file_imgs.close()

