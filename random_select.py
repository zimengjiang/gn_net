# author by LYS 2017/5/24
# for Deep Learning course
'''
1. read the whole files under a certain folder
2. chose 10000 files randomly
3. copy them to another folder and save
'''
import os, random, shutil
 
def copyFile(fileDir,tarDir):
    # 1
	pathDir = os.listdir(fileDir)
    # 2
	sample = random.sample(pathDir, 1000)
	print(sample)
	# 3
	for name in sample:
		shutil.copyfile(fileDir+name, tarDir+name)

def main():
	fileDir = "/home/lechen/gnnet/gn_net_data/cmu/correspondence/"
	tarDir = '/home/lechen/gnnet/gn_net_data_tiny/cmu/correspondence/'
	copyFile(fileDir,tarDir)


if __name__ == '__main__':
    main()
