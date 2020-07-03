import os, random, shutil


def copyFile(fileDir,tarDir):
	pathDir = os.listdir(fileDir)
	sample = random.sample(pathDir, 1000)
	for name in sample:
		shutil.copyfile(fileDir+name, tarDir+name)


def main():
	fileDir = "/home/lechen/gnnet/gn_net_data/cmu/correspondence/"
	tarDir = '/home/lechen/gnnet/gn_net_data_tiny/cmu/correspondence/'
	copyFile(fileDir,tarDir)


if __name__ == '__main__':
    main()
