import os

if __name__ == '__main__':
    path = 'test/dog'
    g = os.walk(path)
    for path,d,filelist in g:
        for filename in filelist:
            name = 'dog'
            os.rename(os.path.join(path,filename),os.path.join(path,name+'.'+filename))
            # print file,'ok'
